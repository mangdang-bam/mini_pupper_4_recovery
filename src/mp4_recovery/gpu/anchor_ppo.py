"""PPO with an anchor: rsl_rl 5.0.1's PPO plus beta x the MSE between the actor's mean and a frozen policy's mean, for
the walker trainer (``gpu/train_walker.py``). The update, the critic warm-up (``actor_warmup_iters``), the
learning-rate bounds and the smoothness terms (CAPS / L2C2: ``caps_temporal``, ``caps_spatial``, ``caps_obs_noise``,
``caps_log``) are those of ``gpu/algorithms.py``; at beta 0 and lambda 0 the update is the stock one.

The anchor network is rebuilt from its state dict (``mlp.*``: any ELU MLP; its input width tells which observation group
it reads: ``anchor_obs_group``, the clean 45-float ``anchor`` group or the actor's own group for a parent with history).

``anchor_mask`` picks the samples the anchor applies to, from each sample's command (the ``anchor_cmd_group`` observation,
(vx, vy, wz) as the policy saw it; None: the anchor group's ``COMMAND_COLS``):
- "all": every sample;
- "nonforward": the omni commands a forward recipe never trained (backward, sideways, diagonals, spins in place;
  ``nonforward_weights``), to hold an omni parent's omni gait while a forward recipe trains;
- "inrange": the commands inside the parent's trained range (``inrange_weights``: forward vx <= ``range_vx``, omni walks
  |v| <= ``range_omni``, |wz| <= ``range_wz``, and a walking command's |wz| <= ``range_walk_wz``): the parent's gait is
  held where it was trained while the wider commands are learned freely.
The masked term is sum_i w_i mse_i / N (beta keeps its per-sample meaning).
"""


from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO

ANCHOR_MASKS = ("all", "nonforward", "inrange")
COMMAND_COLS = slice(6, 9)      # contract.OBS_TERMS: (vx, vy, wz) in the 45-float anchor group (raw, unscaled)
FORWARD_EPS = 0.02              # m/s: |vy| above this, or vx below -this, is not forward walking
SPIN_EPS = 0.1                  # rad/s: a turn in place (|vx|, |vy| <= FORWARD_EPS) above this


def nonforward_weights(command: torch.Tensor, eps: float = FORWARD_EPS, spin_eps: float = SPIN_EPS) -> torch.Tensor:
    """Per sample 1.0 where the (vx, vy, wz) command is an omni command the forward recipe never trained -- backward
    (vx < -eps), sideways or diagonal (|vy| > eps), or a turn in place (|vx|, |vy| <= eps, |wz| > spin_eps) -- else 0.0.
    Standing and slow forward commands are left out: the anchor group sees no terrain, and on a stair tile those
    samples are mid-climb states the flat policy never met."""
    vx, vy, wz = command[..., 0], command[..., 1], command[..., 2]
    spin = (vx.abs() <= eps) & (vy.abs() <= eps) & (wz.abs() > spin_eps)
    return ((vx < -eps) | (vy.abs() > eps) | spin).to(command.dtype)


RANGE_DEFAULT = {"range_vx": 0.12, "range_omni": 0.12, "range_wz": 1.0, "range_walk_wz": 0.6}   # the starting draws


def inrange_weights(command: torch.Tensor, range_vx: float = 0.12, range_omni: float = 0.12, range_wz: float = 1.0,
                    range_walk_wz: float = 0.6, eps: float = FORWARD_EPS) -> torch.Tensor:
    """Per sample 1.0 where the (vx, vy, wz) command lies inside the parent's trained range, else 0.0. Forward (|vy| <=
    eps, vx >= -eps): vx <= ``range_vx``; any other walk: |(vx, vy)| <= ``range_omni``; |wz| <= ``range_wz`` (a turn in
    place); a walking command (|(vx, vy)| > eps) also |wz| <= ``range_walk_wz``. The walker's starting checkpoint was
    trained on forward vx U(0, 0.12) with wz
    U(+-0.6) or the heading loop's (+-0.4), omni walks |v| U(0.05, 0.12) with wz U(+-0.6) or held, spins |wz| U(0.2, 1.0),
    and 10 % standing (``walker_task``'s constants): the defaults."""
    vx, vy, wz = command[..., 0], command[..., 1], command[..., 2]
    speed = torch.hypot(vx, vy)
    fwd = (vy.abs() <= eps) & (vx >= -eps)
    ok = (wz.abs() <= range_wz) & torch.where(fwd, vx <= range_vx, speed <= range_omni)
    ok = ok & ((speed <= eps) | (wz.abs() <= range_walk_wz))
    return ok.to(command.dtype)


def _build_anchor_mlp(state_dict: dict, device: str) -> nn.Sequential:
    """Rebuild a frozen ELU MLP (the actor's mean net) from ``mlp.{i}.{weight,bias}`` and load it strictly."""
    # nn.Linear construction draws from the global torch RNG; building the anchor inside fork_rng keeps the training RNG
    # stream identical to a run without an anchor
    lin = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")})
    with torch.random.fork_rng(devices=[]):
        layers = []
        for j, i in enumerate(lin):
            w = state_dict[f"mlp.{i}.weight"]
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if j < len(lin) - 1:
                layers.append(nn.ELU())
        net = nn.Sequential(*layers)
    mlp_keys = {k.removeprefix("mlp."): v for k, v in state_dict.items() if k.startswith("mlp.")}
    net.load_state_dict(mlp_keys, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def resolve_anchor_path(anchor_path: str) -> Path:
    return Path(anchor_path).expanduser().resolve()


def caps_pair_indices(dones: torch.Tensor) -> torch.Tensor:
    """The flat indices t * N + e (into the storage flattened over (T, N)) of the pairs (t, t + 1) of one env with the
    episode not ended at t: ``dones`` (T, N[, 1]), dones[t, e] the flag stored with step t's transition (the episode
    ended after step t's action, so o_{t+1} is a reset observation). t runs to T - 2; the partner is index + N."""
    d = dones.reshape(dones.shape[0], -1)
    valid = d[:-1] == 0
    return torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)


def actor_mean_from_latent(actor, latent: torch.Tensor) -> torch.Tensor:
    """The actor's deterministic (mean) output from its normalised input (rsl_rl ``MLPModel``: ``get_latent`` then
    ``mlp``, then the distribution's ``deterministic_output``)."""
    out = actor.mlp(latent)
    return actor.distribution.deterministic_output(out) if actor.distribution is not None else out


def caps_temporal_loss(actor, flat_obs, idx: torch.Tensor, num_envs: int) -> torch.Tensor:
    """mean over the pairs ``idx`` (``caps_pair_indices``) of ||mu(o[idx + num_envs]) - mu(o[idx])||^2, the squared
    norm summed over the actions; ``flat_obs`` the storage's observations flattened over (T, N)."""
    lat = actor.get_latent(flat_obs[torch.cat([idx, idx + num_envs])])
    mu0, mu1 = actor_mean_from_latent(actor, lat).chunk(2)
    return (mu1 - mu0).pow(2).sum(dim=-1).mean()


def caps_spatial_loss(actor, obs, clean_mu: torch.Tensor, sigma: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """mean over the samples of ||clean_mu - mu(z + eps)||^2 (summed over the actions), z = the actor's normalised input
    of ``obs``, eps ~ N(0, sigma^2) per input float."""
    lat = actor.get_latent(obs)
    eps = torch.randn(lat.shape, generator=generator, device=lat.device, dtype=lat.dtype)
    return (clean_mu - actor_mean_from_latent(actor, lat + sigma * eps)).pow(2).sum(dim=-1).mean()


class AnchorPPO(PPO):
    """PPO with an action-mean MSE anchor toward a frozen clone."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        anchor_path: str | None,
        anchor_sha256: str | None,
        anchor_beta0: float,
        anchor_hold_iters: int,
        anchor_decay_iters: int,
        anchor_beta_floor: float = 0.0,
        anchor_obs_group: str = "anchor",
        actor_warmup_iters: int = 0,
        anchor_mask: str = "all",
        anchor_push_mask: float | None = None,
        anchor_push_group: str = "push_t",
        anchor_cmd_group: str | None = None,
        anchor_range: dict | None = None,
        caps_temporal: float = 0.0,
        caps_spatial: float = 0.0,
        caps_obs_noise: float = 0.0,
        caps_log: bool = False,
        caps_seed: int = 0,
        lr_min: float = 1.0e-5,
        lr_max: float = 1.0e-2,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("AnchorPPO does not support RND/symmetry (untested interaction).")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("AnchorPPO supports the MLP (non-recurrent) lane only.")
        if anchor_beta0 < 0.0:
            raise ValueError(f"anchor_beta0={anchor_beta0} < 0")
        if anchor_mask not in ANCHOR_MASKS:
            raise ValueError(f"anchor_mask={anchor_mask!r} not in {ANCHOR_MASKS}")
        if anchor_path is None:
            if anchor_beta0 != 0.0:
                raise ValueError("AnchorPPO without an anchor_path needs anchor_beta0 = 0 (the warm-up only)")
            self.anchor_mlp = None
        else:
            path = resolve_anchor_path(anchor_path)
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            expected = str(anchor_sha256).strip().lower()
            if digest != expected:
                raise ValueError(
                    f"Anchor checkpoint hash mismatch: {path} is {digest[:12]}..., "
                    f"cfg registers {expected[:12]}..."
                )
            loaded = torch.load(path, map_location=self.device, weights_only=False)
            self.anchor_mlp = _build_anchor_mlp(loaded["actor_state_dict"], self.device)
        self.anchor_mask = anchor_mask
        self.anchor_cmd_group = anchor_cmd_group
        self.anchor_range = {**RANGE_DEFAULT, **(anchor_range or {})}
        if set(self.anchor_range) != set(RANGE_DEFAULT):
            raise ValueError(f"anchor_range keys: {sorted(RANGE_DEFAULT)}")
        self.anchor_push_mask = None if anchor_push_mask is None else float(anchor_push_mask)
        self.anchor_push_group = anchor_push_group
        self.anchor_obs_group = anchor_obs_group
        self.anchor_beta0 = float(anchor_beta0)
        self.anchor_hold_iters = int(anchor_hold_iters)
        self.anchor_decay_iters = int(anchor_decay_iters)
        self.anchor_beta_floor = float(anchor_beta_floor)
        self._anchor_update_count = 0
        self.actor_warmup_iters = int(actor_warmup_iters)
        if min(caps_temporal, caps_spatial, caps_obs_noise) < 0.0:
            raise ValueError("caps_temporal, caps_spatial and caps_obs_noise must be >= 0")
        if caps_spatial > 0.0 and caps_obs_noise <= 0.0:
            raise ValueError("caps_spatial > 0 needs caps_obs_noise > 0")
        if not 0.0 < lr_min <= lr_max:
            raise ValueError(f"need 0 < lr_min <= lr_max (got {lr_min}, {lr_max})")
        self.caps_temporal = float(caps_temporal)
        self.caps_spatial = float(caps_spatial)
        self.caps_obs_noise = float(caps_obs_noise)
        self.caps_log = bool(caps_log)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self._caps_gen = None
        if self.caps_temporal > 0.0 or self.caps_spatial > 0.0 or self.caps_log:
            self._caps_gen = torch.Generator(device=self.device)
            self._caps_gen.manual_seed(int(caps_seed))

    # -- schedule ---------------------------------------------------------
    def anchor_beta_at(self, it: int) -> float:
        """beta at update-counter ``it`` (counting from launch)."""
        if it < self.anchor_hold_iters:
            return self.anchor_beta0
        if it < self.anchor_hold_iters + self.anchor_decay_iters:
            frac = (it - self.anchor_hold_iters) / max(self.anchor_decay_iters, 1)
            return max(self.anchor_beta_floor, self.anchor_beta0 * (1.0 - frac))
        return self.anchor_beta_floor

    # -- the anchor term ---------------------------------------------------
    def anchor_terms(self, actor_mean: torch.Tensor, anchor_mu: torch.Tensor, command: torch.Tensor,
                     push_t: torch.Tensor | None = None):
        """(the loss term before beta, the MSE over the anchored samples, the anchored fraction). "all": the mean over
        every sample and joint; "nonforward" and "inrange": sum_i w_i mse_i / N, so beta
        keeps its per-sample meaning (not normalised by the anchored count). With ``anchor_push_mask`` S, ``push_t``
        (N, 1) (the stored time since the last kick) multiplies the weights by (push_t > S), "all" included."""
        push = getattr(self, "anchor_push_mask", None)
        if self.anchor_mask == "all" and push is None:
            mse = (actor_mean - anchor_mu).pow(2).mean()
            return mse, mse, 1.0
        per = (actor_mean - anchor_mu).pow(2).mean(dim=-1)
        w = self.anchor_weights(command, push_t).to(per.dtype)
        n = w.sum()
        term = (w * per).sum() / per.shape[0]
        masked = (w * per).sum() / n if n > 0 else per.new_zeros(())
        return term, masked, float(n / per.shape[0])

    def anchor_weights(self, command: torch.Tensor, push_t: torch.Tensor | None = None) -> torch.Tensor:
        """Per sample (``command``: (N, 3), each sample's (vx, vy, wz)): 1 ("all"), ``nonforward_weights``
        ("nonforward") or ``inrange_weights`` ("inrange"), times (push_t > ``anchor_push_mask``) when that is set."""
        if self.anchor_mask == "nonforward":
            w = nonforward_weights(command)
        elif self.anchor_mask == "inrange":
            w = inrange_weights(command, **self.anchor_range)
        else:
            w = torch.ones(command.shape[0], dtype=command.dtype, device=command.device)
        push = getattr(self, "anchor_push_mask", None)
        if push is not None:
            if push_t is None:
                raise ValueError("anchor_push_mask needs the push_t observation of each sample")
            w = w * (push_t.reshape(-1) > push).to(w.dtype)
        return w

    # -- update: the stock rsl_rl loop plus one guarded insertion -----------
    def update(self) -> dict[str, float]:
        beta = self.anchor_beta_at(self._anchor_update_count)
        warmup = self._anchor_update_count < self.actor_warmup_iters
        self._anchor_update_count += 1

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_anchor_loss = 0.0
        mean_anchor_mse = 0.0
        mean_anchor_frac = 0.0
        mean_push_masked = 0.0
        mean_actor_grad_norm = 0.0
        mean_caps_temporal = 0.0
        mean_caps_spatial = 0.0
        mean_kl = 0.0
        mean_lr = 0.0
        grad_norms = {}

        # the smoothness terms (none at lambda 0 without caps_log: the stock computation)
        do_temporal = not warmup and (self.caps_temporal > 0.0 or self.caps_log)
        do_spatial = not warmup and self.caps_obs_noise > 0.0 and (self.caps_spatial > 0.0 or self.caps_log)
        if do_temporal:
            caps_pairs = caps_pair_indices(self.storage.dones)
            caps_obs = self.storage.observations.select(*self.actor.obs_groups).flatten(0, 1)
            do_temporal = caps_pairs.numel() > 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for k, batch in enumerate(generator):
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive" and not warmup:
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.lr_min, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.lr_max, self.learning_rate * 1.5)
                        # a start rate outside [lr_min, lr_max] (a no-op for one inside: today's runs)
                        self.learning_rate = min(self.lr_max, max(self.lr_min, self.learning_rate))
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                mean_kl += float(kl_mean)
            else:               # the log only
                with torch.inference_mode():
                    mean_kl += float(torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params,
                                                                              distribution_params)))
            mean_lr += float(self.optimizer.param_groups[0]["lr"])

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if warmup:          # value function only: the actor receives no gradient
                loss = self.value_loss_coef * value_loss

            # --- the anchor (skipped exactly at beta == 0, so that path is the
            # stock computation) --------------------------------------------
            if beta != 0.0 and not warmup:
                if self.anchor_obs_group not in batch.observations.keys():
                    raise KeyError(
                        f"AnchorPPO: obs group '{self.anchor_obs_group}' missing from rollout "
                        f"storage (available: {list(batch.observations.keys())}). The environment "
                        "must declare the clean anchor group."
                    )
                anchor_obs = batch.observations[self.anchor_obs_group]
                push_t = None
                if self.anchor_push_mask is not None:
                    if self.anchor_push_group not in batch.observations.keys():
                        raise KeyError(f"AnchorPPO: anchor_push_mask needs the obs group '{self.anchor_push_group}' "
                                       f"in the rollout storage (available: {list(batch.observations.keys())})")
                    push_t = batch.observations[self.anchor_push_group]
                    mean_push_masked += float((push_t <= self.anchor_push_mask).float().mean())
                if self.anchor_cmd_group is not None:
                    command = batch.observations[self.anchor_cmd_group][:, :3]
                else:       # the fallback reads the command at COMMAND_COLS: only valid in the plain 45-float group
                    if anchor_obs.shape[-1] != 45:
                        raise ValueError(f"AnchorPPO: no anchor_cmd_group and the anchor group "
                                         f"'{self.anchor_obs_group}' is {anchor_obs.shape[-1]} wide, not 45: its "
                                         "COMMAND_COLS are not the command")
                    command = anchor_obs[:, COMMAND_COLS]
                with torch.no_grad():
                    anchor_mu = self.anchor_mlp(anchor_obs)
                anchor_term, anchor_mse, anchor_frac = self.anchor_terms(self.actor.output_mean, anchor_mu, command,
                                                                         push_t)
                loss = loss + beta * anchor_term
                mean_anchor_mse += anchor_mse.item()
                mean_anchor_loss += beta * anchor_term.item()
                mean_anchor_frac += anchor_frac

            # --- the smoothness terms on the mean (CAPS / L2C2) --------------
            # caps_log: at the update's first mini-batch the actor-gradient norms of the surrogate and of each raw
            # term are measured too (torch.autograd.grad: the .grad buffers and the update are untouched)
            grad_probe = self.caps_log and k == 0 and not warmup
            probe = {}
            if do_temporal:
                if k % self.num_mini_batches == 0:      # each epoch: every valid pair once, split as the mini-batches
                    perm = torch.randperm(caps_pairs.numel(), generator=self._caps_gen, device=caps_pairs.device)
                    caps_chunks = caps_pairs[perm].tensor_split(self.num_mini_batches)
                idx = caps_chunks[k % self.num_mini_batches]
                if idx.numel() > 0:
                    with torch.set_grad_enabled(self.caps_temporal > 0.0 or grad_probe):
                        t_term = caps_temporal_loss(self.actor, caps_obs, idx, self.storage.num_envs)
                    if self.caps_temporal > 0.0:
                        loss = loss + self.caps_temporal * t_term
                    if grad_probe:
                        probe["grad_caps_temporal"] = t_term
                    mean_caps_temporal += t_term.item()
            if do_spatial:
                clean_mu = self.actor.output_mean[:original_batch_size]
                obs_mb = batch.observations[:original_batch_size]
                with torch.set_grad_enabled(self.caps_spatial > 0.0 or grad_probe):
                    s_term = caps_spatial_loss(self.actor, obs_mb, clean_mu, self.caps_obs_noise, self._caps_gen)
                if self.caps_spatial > 0.0:
                    loss = loss + self.caps_spatial * s_term
                if grad_probe:
                    probe["grad_caps_spatial"] = s_term
                mean_caps_spatial += s_term.item()
            if grad_probe:
                probe["grad_surrogate"] = surrogate_loss
                params = [p for p in self.actor.parameters() if p.requires_grad]
                for name, term in probe.items():
                    gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                    grad_norms[name] = float(torch.sqrt(sum(g.pow(2).sum() for g in gs if g is not None)))

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_actor_grad_norm += float(actor_grad_norm)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_anchor_loss /= num_updates
        mean_anchor_mse /= num_updates
        mean_anchor_frac /= num_updates
        mean_push_masked /= num_updates
        mean_actor_grad_norm /= num_updates
        mean_caps_temporal /= num_updates
        mean_caps_spatial /= num_updates
        mean_kl /= num_updates
        mean_lr /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "anchor": mean_anchor_loss,
            "anchor_mse": mean_anchor_mse,
            "anchor_beta": beta,
            "anchor_frac": mean_anchor_frac,
            "actor_grad_norm": mean_actor_grad_norm,
            "kl": mean_kl,
            "lr_mean": mean_lr,
        }
        if self._caps_gen is not None:      # raw (before lambda); 0 in the warm-up updates
            loss_dict["caps_temporal"] = mean_caps_temporal
            loss_dict["caps_spatial"] = mean_caps_spatial
            loss_dict.update(grad_norms)    # caps_log: the first mini-batch's actor-gradient norms (before clipping)
        if self.anchor_push_mask is not None:      # the share of samples within anchor_push_mask s of a kick
            loss_dict["anchor_push_masked"] = mean_push_masked
        return loss_dict
