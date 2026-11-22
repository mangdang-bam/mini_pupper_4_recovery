#!/usr/bin/env python
"""Fine-tune the walker with PPO on the walker's training env (``walker.task``), anchored to a parent policy
(``anchor_ppo.AnchorPPO``).

    python -m mp4_rl.skills.walker.train --resume mp4_rl/skills/walker/checkpoints/sku2/walker_start/model_6297.pt \\
        --anchor mp4_rl/skills/walker/checkpoints/sku2/walker_start/model_6297.pt --actuator md300 --limits --stand-offset-abd 12 \\
        --attachment-lidar 0.5 --omni-mix spin_up --cmd-full --cmd-ramp 300 --envs 1024 --iters 1000 --seed 1

``mp4_rl/skills/walker/recipes/sku2/step1.sh`` to ``step4.sh`` hold the fine-tuning steps' command lines.

**Resume** (required): actor, critic, std, optimizer and iteration from ``--resume`` (``model_k.pt``; its run's
``config.json`` beside it gives the action scale and the actor's history); learning rate ``--lr0`` (1e-4), the critic
warmed up first (``--critic-warmup``, value-only updates; the anchor's hold starts after them).

**Anchor** (``--anchor``): a checkpoint (``.pt``: its actor) or an exported policy directory (``policy.npz``, rescaled to
this run's action scale). A 45-input network reads the clean ``anchor`` observation group; a network as wide as the actor (a parent with history) reads the actor's own observation. Beta ``--beta0`` held
``--beta-hold`` iterations then fading to 0 over ``--beta-decay``. ``--anchor-mask``: ``inrange`` (default) anchors only
the samples whose command lies inside the parent's trained range (``--anchor-range VX OMNI WZ WALK_WZ``, default
the walker's starting checkpoint's 0.12 m/s forward, 0.12 m/s omni, 1.0 rad/s spins, 0.6 rad/s while walking): the
parent's gait is held there while the wider commands train free (``--anchor-push-mask S`` also frees the samples within
S s of a kick); ``nonforward`` anchors only the omni commands a forward-only recipe never trains (backward, sideways,
diagonals, spins in place); ``all`` every sample. The mask reads each sample's command from the ``cmd`` group.

**Commands** (``walker.task.RelativeHeadingCommand``): the forward draw U(0, ``--vx-max``) on every column, the heading
loop (``--heading-law``, ``--heading-wz``), and on the non-stair columns the omni mix ``--omni-mix`` (or five shares
``--omni-shares F O S T W``: forward, omni walk, spin, stand, walk-and-turn; the mix ``default`` is 0.40 0.40 0.10 0.10 0,
``spin_up`` 0.25 0.30 0.25 0.10 0.10, more turning in place). Range curriculum: ``--cmd-full`` sets the end ranges to the
classic controller's (forward and omni 0.20 m/s, spins 2.0 rad/s, walk-and-turn 1.0 rad/s, omni walks' yaw 1.0 rad/s),
each also settable (``--fwd-vx-end``, ``--omni-speed-end``, ``--spin-wz-end``, ``--turn-wz-end``,
``--omni-wz-end``); the start values are the parent's (0.12, 0.12, 1.0, 0.6, 0.6); they move linearly over
``--cmd-ramp N`` iterations from this run's first (0: the end ranges at once). The stair columns keep ``--vx-max``.

**Plant:** ``--actuator md01|md300`` (+ ``--supply-v``, ``--supply-drop-ohm``, ``--r-scale``), ``--legs`` (sku_2),
``--foot``, ``--foot-radius``, ``--masses-json`` (a file name in the robot's measured data in mp4_sim, or a path),
``--limits`` (walk mode), ``--stand-offset-abd`` (the abductions' stand shifted outward; the recipes use 12 deg, which the
run's ``config.json`` records and ``common.export`` writes into the policy's meta), ``--attachment H,R,X,M`` (every robot)
or ``--attachment-lidar P`` (this robot's lidar 35 mm high, 19 mm radius, 43 mm forward, 43.2 g, present on a share P of
the robots), kicks ``--push-mps LO HI --push-interval A B`` (and ``--topple-mps`` / ``--topple-interval``).

The terrain and reward recipe defaults are the walker's: narrow terrain, risers 25-45 mm, an episode on stairs ending as a
success once the robot is off them (``--stair-exit``), foot softness 5-20 ms, friction 0.4-1.3, apex clearance -2, stumble
-0.5, progress 2, overspeed -2, tilt limit 1.0, robots starting on terrain levels up to 9 (``--max-init-level``), the PI
heading loop at 0.4 rad/s, and an omni walk moving its robot up a terrain level once it covers 0.8 of its commanded path
or half a tile, whichever is shorter (``--omni-promote-rel``). ``config.json`` records the argv, the repository commit and
dirty flag, the parent's and the anchor's hashes, the masses and the built env (total mass, legs, actuator, attachment,
contract, decoder clamps).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from mp4_sim import robots
from mp4_sim.actuators import choice as servo_choice
from mp4_sim.envs.attachment import Attachment

from mp4_rl import paths
from mp4_rl.common.bounds import add_legs_args, eval_legs, legs_flags
from mp4_rl.paths import sha256

STEPS_PER_ITER = 24
PREV_ACTION = slice(33, 45)      # policy.OBS_TERMS: the previous action in the 45-float observation
LIDAR = Attachment((35.0, 35.0), (43.2, 43.2), (43.0, 43.0), 19.0)    # the robot's lidar, LDROBOT STL-06P
CLASSIC = {"fwd_vx_end": 0.20, "omni_speed_end": 0.20, "spin_wz_end": 2.0, "turn_wz_end": 1.0, "omni_wz_end": 1.0}
PARENT = {"fwd_vx": 0.12, "omni_speed": 0.12, "spin_wz": 1.0, "turn_wz": 0.6, "omni_wz": 0.6}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", required=True, help="the parent's model_<k>.pt (its run's config.json beside it)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--out", default=str(paths.RUNS / "walker"))
    ap.add_argument("--tag", default="")
    # the anchor
    ap.add_argument("--anchor", help="a checkpoint (.pt) or a policy directory (policy.npz) to anchor to")
    ap.add_argument("--anchor-mask", choices=("all", "nonforward", "inrange"), default="inrange")
    ap.add_argument("--anchor-range", type=float, nargs=4, metavar=("VX", "OMNI", "WZ", "WALK_WZ"),
                    default=(0.12, 0.12, 1.0, 0.6),
                    help="inrange: the parent's trained range (m/s forward, m/s omni, rad/s spin, rad/s while walking)")
    ap.add_argument("--anchor-group", choices=("auto", "anchor", "actor"), default="auto",
                    help="the observation the anchor reads (auto: by its input width)")
    ap.add_argument("--beta0", type=float, default=5.0)
    ap.add_argument("--beta-hold", type=int, default=100000)
    ap.add_argument("--beta-decay", type=int, default=0)
    ap.add_argument("--critic-warmup", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=1.0e-4)
    ap.add_argument("--lr-max", type=float, default=1.0e-2, help="the adaptive learning rate's ceiling (AnchorPPO lr_max)")
    ap.add_argument("--entropy-coef", type=float, default=0.005)
    # the plant
    servo_choice.add_args(ap)
    add_legs_args(ap, "sku_2")
    ap.add_argument("--masses-json", help="the plant's masses (a file name in the robot's measured data in mp4_sim or a "
                                          "path; default the robot's own weighed masses)")
    ap.add_argument("--limits", action="store_true", help="the measured joint limits, walk mode (model and decoder)")
    ap.add_argument("--allow-limits-contract-mismatch", action="store_true",
                    help="with --limits: train even though the packaged limits were traced at another contract than "
                         "the one in force")
    ap.add_argument("--stand-offset-abd", type=float, default=0.0, metavar="DEG",
                    help="the abductions' stand shifted outward (the walker's 12)")
    ap.add_argument("--attachment", type=Attachment.parse, metavar="HEIGHT_MM,RADIUS_MM,X_M,MASS_G",
                    help="a fixed attachment on every robot")
    ap.add_argument("--attachment-lidar", type=float, metavar="P",
                    help="the robot's lidar (35 mm, r 19 mm; SKU2 x 43 mm, 43.2 g; SKU1 x -5 mm, 47.2 g with its mount) "
                         "present on a share P of the robots")
    ap.add_argument("--attachment-lidar-x", type=float, nargs=2, metavar=("LO", "HI"),
                    help="the lidar's axis drawn per robot, forward of the base origin (mm; default the robot's)")
    ap.add_argument("--attachment-lidar-y", type=float, nargs=2, metavar=("LO", "HI"),
                    help="the lidar's axis drawn per robot, left of the mid-plane (mm; default 0)")
    ap.add_argument("--com-height-mm", type=float, nargs=2, metavar=("LO", "HI"),
                    help="the whole robot's centre-of-mass height above the floor at the stand (mm, as a weighing on "
                         "scales gives it), drawn per robot: one shared z shift of the base and the four hip module "
                         "bodies, after the attachment (mp4_sim.envs.com_height; default off)")
    ap.add_argument("--leg-leg-contacts", action="store_true",
                    help="the get-up's self-collision on the walker's plant: leg <-> leg, leg <-> hip can, thigh <-> base "
                         "unless --limits (mp4_sim.envs.contact; default off)")
    ap.add_argument("--push-mps", type=float, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--push-interval", type=float, nargs=2, metavar=("A", "B"))
    ap.add_argument("--topple-mps", type=float, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--topple-interval", type=float, nargs=2, metavar=("A", "B"))
    ap.add_argument("--topple-start-iter", type=int, default=0, metavar="N",
                    help="tier 2 kicks only from iteration N after the resumed one (before: none)")
    ap.add_argument("--save-at", type=int, nargs="*", default=[300], metavar="N",
                    help="also save model_<resumed iteration + N> after these iterations of this run (besides every 50)")
    ap.add_argument("--push-fall-grace", type=float, metavar="S", help="default 2 with a kick tier")
    ap.add_argument("--anchor-push-mask", type=float, metavar="S",
                    help="no anchor on the samples within S s of a kick (needs an anchor and a kick tier)")
    # the terrain and the rewards (the walker's defaults)
    ap.add_argument("--terrain", choices=("narrow", "rough", "flat"), default="narrow")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--narrow-risers", type=float, nargs=2, default=(0.025, 0.045))
    ap.add_argument("--boxes-share", type=float)
    ap.add_argument("--max-init-level", type=int, default=9)
    ap.add_argument("--stair-exit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--foot-softness", type=float, nargs=2, default=(0.005, 0.02))
    ap.add_argument("--friction", type=float, nargs=2, default=(0.4, 1.3))
    ap.add_argument("--tilt-limit", type=float, default=1.0)
    ap.add_argument("--clearance", choices=("speed", "apex"), default="apex")
    ap.add_argument("--clearance-weight", type=float, default=-2.0)
    ap.add_argument("--clearance-target", type=float, default=0.025)
    ap.add_argument("--stumble", type=float, default=-0.5)
    ap.add_argument("--progress", type=float, default=2.0)
    ap.add_argument("--overspeed", type=float, default=-2.0)
    ap.add_argument("--omni-promote-rel", type=float, default=0.8)
    ap.add_argument("--leg-contact", type=float, default=-0.5)
    ap.add_argument("--abduction", type=float, default=0.0)
    ap.add_argument("--heading-law", choices=("p", "pi"), default="pi")
    ap.add_argument("--heading-wz", type=float, default=0.4)
    # the commands
    ap.add_argument("--vx-max", type=float, default=0.12, help="the forward draw's top (stair columns keep it)")
    from mp4_rl.skills.walker.task import OMNI_MIXES
    ap.add_argument("--omni-mix", choices=sorted(OMNI_MIXES), default="default")
    ap.add_argument("--omni-shares", type=float, nargs=5, metavar=("F", "O", "S", "T", "W"),
                    help="forward, omni walk, spin, stand, walk-and-turn shares (override the mix key's)")
    ap.add_argument("--cmd-full", action="store_true",
                    help="the end ranges are the classic controller's (0.20 m/s, 2.0 rad/s spins)")
    ap.add_argument("--fwd-vx-end", type=float)
    ap.add_argument("--omni-speed-end", type=float)
    ap.add_argument("--spin-wz-end", type=float)
    ap.add_argument("--turn-wz-end", type=float)
    ap.add_argument("--omni-wz-end", type=float)
    ap.add_argument("--cmd-fast-share", type=float, default=0.0, metavar="P",
                    help="with probability P a walk's speed or a spin's |wz| (omni columns) is drawn from the fast band "
                         "instead of the full range (logged as Curriculum/command_range/fast_frac)")
    ap.add_argument("--cmd-fast-speed", type=float, nargs=2, default=(0.15, 0.20), metavar=("LO", "HI"),
                    help="the fast band of forward, omni and walk-and-turn speeds (m/s)")
    ap.add_argument("--cmd-fast-wz", type=float, nargs=2, default=(1.5, 2.0), metavar=("LO", "HI"),
                    help="the fast band of spins' |wz| (rad/s)")
    ap.add_argument("--yaw-zero-share", type=float, default=0.0, metavar="Q",
                    help="a share Q of the omni walks not held by the heading loop are commanded with wz = 0 exactly")
    ap.add_argument("--yaw-hold-weight", type=float, default=0.0, metavar="W",
                    help="weight of yaw_hold: exp(-((w - wz_cmd) / --yaw-hold-std)^2) on walks with |wz_cmd| <= 0.05")
    ap.add_argument("--yaw-hold-std", type=float, default=0.10, metavar="RAD_S")
    ap.add_argument("--spin-progress-weight", type=float, default=0.0, metavar="W",
                    help="weight of spin_progress: clip(w / wz_cmd, 0, 1) on spins and walk-and-turns with |wz_cmd| > "
                         "0.3 (a partial turn pays; freezing or the wrong way 0)")
    ap.add_argument("--cmd-ramp", type=int, default=0, metavar="N",
                    help="iterations from the parent's ranges to the end ranges (0: the end ranges at once)")
    return ap


def command_ranges(a) -> dict:
    """The curriculum's (start, end) per range: start = the parent's (``PARENT``; the forward draw's ``--vx-max``), end
    = the flag, else ``CLASSIC`` with ``--cmd-full``, else the start."""
    out = {}
    for key, flag, start in (("fwd_vx_range", "fwd_vx_end", a.vx_max), ("omni_speed_max", "omni_speed_end",
                                                                          PARENT["omni_speed"]),
                             ("spin_wz_max", "spin_wz_end", PARENT["spin_wz"]), ("turn_wz_max", "turn_wz_end",
                                                                                 PARENT["turn_wz"]),
                             ("omni_wz", "omni_wz_end", PARENT["omni_wz"])):
        end = getattr(a, flag)
        if end is None:
            end = CLASSIC[flag] if a.cmd_full else start
        if end < start:
            raise SystemExit(f"--{flag.replace('_', '-')} {end} is below its start {start}")
        out[key] = (float(start), float(end))
    return out


def attachment_of(a) -> Attachment | None:
    if a.attachment is not None and a.attachment_lidar is not None:
        raise SystemExit("--attachment and --attachment-lidar exclude each other")
    lx, ly = getattr(a, "attachment_lidar_x", None), getattr(a, "attachment_lidar_y", None)
    if (lx is not None or ly is not None) and a.attachment_lidar is None:
        raise SystemExit("--attachment-lidar-x / -y need --attachment-lidar")
    if a.attachment_lidar is not None:
        if not 0.0 <= a.attachment_lidar <= 1.0:
            raise SystemExit("--attachment-lidar P: a share in [0, 1]")
        x, mass = LIDAR.x_mm, LIDAR.mass_g
        if (getattr(a, "robot", None) or robots.DEFAULT) != robots.DEFAULT:       # the robot's own lidar mount
            r = robots.get(a.robot)
            x, mass = (r.lidar_x_mm,) * 2, (r.lidar_mass_g,) * 2
        if lx is not None:
            x = tuple(lx)
        kw = {} if ly is None else {"y_mm": tuple(ly)}
        return Attachment(LIDAR.height_mm, mass, x, LIDAR.radius_mm, present=float(a.attachment_lidar), **kw)
    return a.attachment


def walker_options(a, src_opts: dict, start_step: int):
    """The ``WalkerOptions`` of a parsed command line, the parent's run options ``src_opts`` (action scale, history)
    and the resumed step (the curriculum's start)."""
    from mp4_rl.skills.walker.task import WalkerOptions, mix_shares
    for x, y in (("push_mps", "push_interval"), ("topple_mps", "topple_interval")):
        if (getattr(a, x) is None) != (getattr(a, y) is None):
            raise SystemExit(f"--{x.replace('_', '-')} and --{y.replace('_', '-')} go together")
    kicks = a.push_mps is not None or a.topple_mps is not None
    if a.push_fall_grace and not kicks:
        raise SystemExit("--push-fall-grace needs a kick tier")
    for name in ("cmd_fast_share", "yaw_zero_share"):
        if not 0.0 <= getattr(a, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')}: a share in [0, 1]")
    for name in ("cmd_fast_speed", "cmd_fast_wz"):
        lo, hi = getattr(a, name)
        if not 0.0 <= lo <= hi:
            raise SystemExit(f"--{name.replace('_', '-')}: LO <= HI, >= 0")
    if a.yaw_hold_std <= 0:
        raise SystemExit("--yaw-hold-std > 0")
    if a.topple_start_iter and a.topple_mps is None:
        raise SystemExit("--topple-start-iter needs --topple-mps")
    if a.anchor_push_mask is not None and not (kicks and a.anchor):
        raise SystemExit("--anchor-push-mask needs an anchor and a kick tier")
    if a.anchor is None and a.beta0:
        print("NOTE no --anchor: plain PPO with the critic warm-up", flush=True)
    opt = WalkerOptions(
        num_envs=a.envs, noisy=True, noise_ramp=True, randomise=True,
        action_scale=tuple(src_opts["action_scale"]), history=int(src_opts.get("history", 0) or 0),
        **({"masses_json": a.masses_json} if a.masses_json else {}),
        terrain=a.terrain, rows=a.rows, terrain_seed=a.terrain_seed, max_init_level=a.max_init_level,
        narrow_risers=tuple(a.narrow_risers) if a.terrain == "narrow" else None, boxes_share=a.boxes_share,
        friction_range=tuple(a.friction), foot_softness=tuple(a.foot_softness), tilt_limit=a.tilt_limit,
        stair_exit=bool(a.stair_exit) and a.terrain == "narrow", clearance=a.clearance,
        clearance_weight=a.clearance_weight, clearance_target=a.clearance_target, stumble=a.stumble,
        progress=a.progress, overspeed=a.overspeed, omni_promote_rel=a.omni_promote_rel, leg_contact=a.leg_contact,
        abduction=a.abduction, heading="loop", heading_law=a.heading_law, heading_wz=a.heading_wz, vx_max=a.vx_max,
        omni_mix=a.omni_mix, omni_shares=tuple(a.omni_shares) if a.omni_shares else None,
        cmd_ramp_iters=a.cmd_ramp, cmd_ramp_start_step=start_step, **command_ranges(a),
        cmd_fast_share=a.cmd_fast_share, fast_speed=tuple(a.cmd_fast_speed), fast_wz=tuple(a.cmd_fast_wz),
        yaw_zero_share=a.yaw_zero_share, yaw_hold_weight=a.yaw_hold_weight, yaw_hold_std=a.yaw_hold_std,
        spin_progress_weight=a.spin_progress_weight,
        push_range=tuple(a.push_mps) if a.push_mps else None,
        push_interval=tuple(a.push_interval) if a.push_mps else None,
        topple_range=tuple(a.topple_mps) if a.topple_mps else None,
        topple_interval=tuple(a.topple_interval) if a.topple_mps else None,
        topple_start_step=start_step + STEPS_PER_ITER * a.topple_start_iter,
        push_fall_grace=a.push_fall_grace if a.push_fall_grace is not None else (2.0 if kicks else 0.0),
        push_obs=a.anchor_push_mask is not None,
        limits=a.limits, stand_offset_abd_deg=a.stand_offset_abd, attachment=attachment_of(a),
        com_height_mm=tuple(a.com_height_mm) if getattr(a, "com_height_mm", None) else None,
        **servo_choice.options_from_args(a),
        **eval_legs({}, **legs_flags(a), leg_leg_contacts=bool(getattr(a, "leg_leg_contacts", False))))
    mix_shares(opt)             # validated
    return opt


def limits_guard(a) -> dict | None:
    """With ``--limits``: the packaged limits' contract against the contract in force (``limits.limits_contract_check``);
    a mismatch stops the run unless ``--allow-limits-contract-mismatch``. None without ``--limits``."""
    if not a.limits:
        return None
    from mp4_rl.common.bounds import limits_contract_check
    try:
        return limits_contract_check(a.allow_limits_contract_mismatch, robot=getattr(a, "robot", None))
    except ValueError as e:
        raise SystemExit(f"{e} (--allow-limits-contract-mismatch)")


def runner_cfg(a, alg_extra: dict) -> dict:
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    cfg = asdict(RslRlOnPolicyRunnerCfg(
        seed=a.seed, num_steps_per_env=STEPS_PER_ITER, max_iterations=a.iters, save_interval=50,
        experiment_name="walker", logger="tensorboard", clip_actions=1.0,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        actor=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=a.entropy_coef,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=a.lr0, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0, class_name="mp4_rl.common.anchor_ppo:AnchorPPO")))
    cfg["algorithm"].update(lr_min=1.0e-5, lr_max=a.lr_max)
    cfg["algorithm"].update(alg_extra)
    return cfg


def npz_to_checkpoint(policy_dir: Path, new_scale, out: Path) -> Path:
    """An exported policy as an actor checkpoint in this run's action scale: output rows times k = old / new per joint and
    the previous-action input columns divided by k (every history frame), so the same state gives the same target."""
    import torch
    d = np.load(policy_dir / "policy.npz", allow_pickle=False)
    n = int(d["n_layers"])
    k = np.asarray(d["scale"], dtype=np.float32) / np.asarray(new_scale, dtype=np.float32)
    meta = json.loads(str(d["meta"]))
    hist = max(int(meta.get("history", 0) or 0), 1)
    sd = {}
    for i in range(n):
        w, b = d[f"W{i}"].astype(np.float32), d[f"b{i}"].astype(np.float32)
        if i == 0:
            w = w.copy()
            if hist == 1:
                w[:, PREV_ACTION] = w[:, PREV_ACTION] / k[None, :]
            else:       # stacked: each term's frames oldest to newest; prev_action is the last term (12 x hist)
                start = 33 * hist
                w[:, start:start + 12 * hist] = w[:, start:start + 12 * hist] / np.tile(k, hist)[None, :]
        if i == n - 1:
            w, b = w * k[:, None], b * k
        sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"] = torch.tensor(w), torch.tensor(b)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": sd, "source": str(policy_dir), "rescale": k.tolist()}, out)
    return out


def anchor_setup(a, log_dir: Path, action_scale, actor_width: int, src_scale) -> dict:
    """AnchorPPO's anchor keywords (and the anchor network's checkpoint in ``log_dir``)."""
    import torch
    base = dict(anchor_beta_floor=0.0, actor_warmup_iters=a.critic_warmup, anchor_cmd_group="cmd")
    if a.anchor is None:
        return dict(base, anchor_path=None, anchor_sha256=None, anchor_beta0=0.0, anchor_hold_iters=0,
                    anchor_decay_iters=0, anchor_obs_group="anchor", anchor_mask="all")
    src = Path(a.anchor).resolve()
    if src.is_dir():
        ck = npz_to_checkpoint(src, action_scale, log_dir / "anchor_rescaled.pt")
    else:
        if not np.allclose(src_scale, action_scale):
            raise SystemExit("a checkpoint anchor must share the run's action scale")
        sd = torch.load(src, map_location="cpu", weights_only=False)["actor_state_dict"]
        if any(k.startswith("obs_normalizer.") for k in sd):
            raise SystemExit("--anchor: a checkpoint with an observation normaliser is not supported")
        ck = log_dir / "anchor_actor.pt"
        torch.save({"actor_state_dict": {k: v for k, v in sd.items() if k.startswith("mlp.")}, "source": str(src)}, ck)
    width = int(torch.load(ck, map_location="cpu", weights_only=False)["actor_state_dict"]["mlp.0.weight"].shape[1])
    group = a.anchor_group
    if group == "auto":
        group = "anchor" if width == 45 else "actor" if width == actor_width else None
    want = 45 if group == "anchor" else actor_width
    if group is None or width != want:
        raise SystemExit(f"--anchor reads {width} inputs: neither the clean 45-float group nor the actor's {actor_width}")
    vx, omni, wz, walk = a.anchor_range
    if a.anchor_push_mask is not None:
        base.update(anchor_push_mask=a.anchor_push_mask, anchor_push_group="push_t")
    return dict(base, anchor_path=str(ck), anchor_sha256=sha256(ck), anchor_beta0=a.beta0,
                anchor_hold_iters=a.critic_warmup + a.beta_hold, anchor_decay_iters=a.beta_decay,
                anchor_obs_group=group, anchor_mask=a.anchor_mask,
                anchor_range={"range_vx": vx, "range_omni": omni, "range_wz": wz, "range_walk_wz": walk})


def resume_step(path) -> int:
    """The env step a checkpoint resumes at (mjlab's runner restores the env's step counter from it)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    s0 = ((ck.get("infos") or {}).get("env_state") or {}).get("common_step_counter")
    return int(s0) if s0 is not None else STEPS_PER_ITER * int(ck.get("iter", 0))


def built_env(env, opt) -> dict:
    """What the built env runs: the total mass per robot (min, max), legs, servo model, attachment, the robot's model,
    calibration and limits files, clamps."""
    from mp4_sim.robot.model import legs_of

    from mp4_rl.common import built_env as built
    from mp4_rl.common.mdp import servo_action
    lg = legs_of(opt)
    bm = env.sim.model.body_mass
    bm = bm.detach().cpu().numpy() if hasattr(bm, "detach") else np.asarray(bm)
    tot = bm.reshape(-1, bm.shape[-1]).sum(axis=-1)
    s = servo_action(env)
    out = {"sum_body_mass_kg": {"min": round(float(tot.min()), 5), "max": round(float(tot.max()), 5),
                                "mean": round(float(tot.mean()), 5)},
           "sum_body_mass_mj_model_kg": round(float(env.sim.mj_model.body_mass.sum()), 5),
           "legs": eval_legs({}, opt.legs, opt.foot, opt.foot_radius_m, opt.robot, opt.leg_leg_contacts,
                             opt.display_collision),
           "actuator": opt.actuator, "supply_v_range": opt.supply_v_range,
           "supply_drop_ohm_range": opt.supply_drop_ohm_range, "r_scale_range": opt.r_scale_range,
           "attachment": asdict(opt.attachment) if opt.attachment else None,
           **built.files_record(lg.robot, lg),
           "stand_offset_abd_deg": float(s.cfg.stand_offset_abd_deg),
           "walk_bounds_deg": None if s.bounds is None else {
               "lo": [round(float(v), 2) for v in np.degrees(s.bounds.lo.detach().cpu().numpy()).ravel()[:12]],
               "hi": [round(float(v), 2) for v in np.degrees(s.bounds.hi.detach().cpu().numpy()).ravel()[:12]]}}
    draw = getattr(env, "attachment_draw", None)
    if draw is not None:
        h = draw["height_m"].detach().cpu().numpy()
        out["attachment_present_share"] = round(float((h > 0).mean()), 4)
    return out


def save_extra(runner, log_dir: Path, iters: set) -> None:
    """Also save ``model_<it>.pt`` after the update of each iteration in ``iters`` (rsl_rl saves only every
    ``save_interval``): the runner's per-iteration log call is wrapped."""
    log0 = runner.logger.log

    def log(*args, **kw):
        out = log0(*args, **kw)
        it = kw.get("it")
        if it in iters and not (log_dir / f"model_{it}.pt").exists():
            runner.save(str(log_dir / f"model_{it}.pt"))
        return out

    runner.logger.log = log


def main(argv=None):
    a = build_parser().parse_args(argv)
    import mp4_sim
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    import mp4_rl
    from mp4_rl.common.env import describe
    from mp4_rl.skills.walker.task import make_walker_env_cfg
    print("mp4_rl", mp4_rl.__file__, "mp4_sim", mp4_sim.__version__, mp4_sim.__file__, flush=True)

    guard = limits_guard(a)
    print("limits contract", json.dumps(guard), flush=True)
    resume = Path(a.resume).resolve()
    src = json.loads((resume.parent / "config.json").read_text())
    src_opts = src["task"]["options"]
    start = resume_step(resume)
    opt = walker_options(a, src_opts, start)
    env_cfg = make_walker_env_cfg(opt)
    env_cfg.seed = a.seed
    log_dir = Path(a.out) / (f"{datetime.now():%Y%m%d_%H%M%S}_s{a.seed}" + (f"_{a.tag}" if a.tag else ""))
    log_dir.mkdir(parents=True, exist_ok=True)

    actor_width = 45 * max(opt.history, 1)
    alg = anchor_setup(a, log_dir, opt.action_scale, actor_width, src_opts["action_scale"])
    cfg = runner_cfg(a, alg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=1.0), cfg, str(log_dir), "cuda:0")
    if not np.allclose(src_opts["action_scale"], opt.action_scale):
        raise SystemExit("the parent's action scale differs (a resume keeps the action scale)")
    runner.load(str(resume), map_location="cuda:0",
                load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True})

    env_rec = built_env(env, opt)
    env_rec["limits_contract"] = guard
    print("built env", json.dumps(env_rec), flush=True)
    from mp4_sim import paths as sim_paths

    from mp4_rl.common import built_env as built
    ver = built.run_versions()
    mpath = built.masses_path(opt.masses_json, opt.robot)
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "walker_task": True, "runner": cfg,
        "algorithm_class": f"{type(runner.alg).__module__}.{type(runner.alg).__name__}", "noise": "measured",
        "resume": str(resume), "resume_sha256": sha256(resume), "resume_step": start,
        "parent_config": {"argv": src.get("argv"), "repo_commit": src.get("repo_commit")},
        "anchor": a.anchor, "anchor_sha256_source": (sha256(Path(a.anchor) / "policy.npz") if a.anchor and
                                                     Path(a.anchor).is_dir() else sha256(Path(a.anchor)) if a.anchor
                                                     else None),
        "masses_json": sim_paths.name(mpath), "masses_sha256": sha256(mpath), "masses": json.loads(mpath.read_text()),
        "built_env": env_rec, "critic_warmup": a.critic_warmup, "lr0": a.lr0,
        "repo_commit": ver["mp4_rl"]["commit"], "repo_dirty": ver["mp4_rl"]["dirty"], "versions": ver,
        "argv": sys.argv},
        indent=1, default=str))
    # the runner's iteration (the checkpoint's; the env step counter may run ahead of 24 x it after earlier resumes)
    save_extra(runner, log_dir, {int(runner.current_learning_iteration) + n for n in a.save_at})
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    print("RUN_DIR", log_dir)
    del torch


if __name__ == "__main__":
    main()
