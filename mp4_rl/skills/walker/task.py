"""The walker's training environment: the walking task (``common.env``: this package's plant, servo models, legs,
masses, limits, stance offset) with the walker's training recipe: its terrain, commands and rewards. Read by
``walker/train.py``.

What it adds to the evaluation env (``common.env.make_terrain_env_cfg``):
- **terrain:** the ``narrow`` mix (``walker_terrain``: rough families, jittered pits and pyramids, narrow straight flights
  with risers ``narrow_risers``) or the rough mix, a curriculum on the progress walked (``terrain_levels_path``); with
  ``stair_exit`` a stair tile ends as a success once the robot is off its stairs, a narrow flight as a failure when a
  foot steps off a side (``side_exit``);
- **commands** (``RelativeHeadingCommand``): forward vx U(0, ``vx_max``), wz U(+-``wz_max``), 10 % standing, resampled
  every 5-10 s; with ``heading="loop"`` most envs take wz from the robot's heading loop (``heading_law`` "pi": the PI law,
  clipped at ``heading_wz``); narrow flights hold their spawn heading. ``omni_mix``: on the non-stair columns each
  resample draws forward / omni walk / spin in place / standing / walk-and-turn (``OMNI_MIXES`` or ``omni_shares``).
  The ranges of the forward draw (on the omni columns), the omni walks' speed, their yaw rate, the spins' and the
  walk-and-turns' yaw rate move linearly from their start to their end values over ``cmd_ramp_iters`` iterations from
  ``cmd_ramp_start_step`` (the range curriculum; start = end: no curriculum, the plain draws exactly);
- **rewards:** the walker's set: velocity and yaw-rate tracking, the regularisers, uprightness against the terrain normal, swing
  apex clearance (or a speed-weighted clearance), progress along the command, overspeed on omni walks, stumble,
  air time, leg contacts, stand still; a zero-weight path tracker for the curriculum;
- **critic:** a height scan around the body and each foot's height over the terrain;
- **observation groups** beside the actor and the critic: ``anchor`` (the clean 45 floats, for an anchor that reads no
  history) and ``cmd`` (the command, read by ``AnchorPPO``'s command mask); no network reads them;
- **disturbances:** mjlab's 0.05 m/s box push or ``PushKick`` tiers (``push_range`` / ``topple_range``);
- **plant draws:** friction, base mass x U(0.9, 1.1), the feet's contact softness, and an ``attachment`` (``mp4_sim.envs.attachment``,
  per robot; with ``present`` a share of robots carry it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mp4_sim.envs import terrain as terrain_mod
from mp4_sim.envs.attachment import Attachment
from mp4_sim.robot.model import FOOT_GEOMS, legs_of, robot_z_shift

from mp4_rl.common import mdp
from mp4_rl.common.env import (
    TerrainOptions,
    _foot_site_ids,
    env_and_scene,
    foot_heights_centre,
    foot_r,
    late_events_last,
    out_of_bounds,
    servo_limits,
)
from mp4_rl.common.policy import ABDUCTION
from mp4_rl.skills.walker import terrain as wt

# ---------------------------------------------------------------------------------------------------- commands
# Shares per resample on the non-stair columns: forward (the forward draw, never standing), omni walk, spin in place,
# standing, walk-and-turn. "default": the walker's starting mix (four shares: no walk-and-turn draw, so the random
# stream is the four-mode one).
OMNI_MIXES = {"default": (0.40, 0.40, 0.10, 0.10, 0.0),
              "spin_up": (0.25, 0.30, 0.25, 0.10, 0.10)}
OMNI_SPEED = (0.05, 0.12)       # m/s: an omni walk's speed, at a uniform angle in (-pi, pi]
OMNI_HEADING_SHARE = 0.5        # of omni walks: heading envs holding the heading they had (with heading="loop")
OMNI_WZ = 0.6                   # rad/s: the other omni walks' yaw rate, U(+- this)
SPIN_WZ = (0.2, 1.0)            # rad/s: a spin's |wz|, random sign, vx = vy = 0
TURN_WZ = (0.2, 0.6)            # rad/s: a walk-and-turn's |wz|, random sign (the omni walk's speed and angle)
OMNI_MIN_CMD = 0.02             # m/s: PathTracker (omni): below this commanded speed nothing is walked
OMNI_CLEAR_GRID = (0.03, 0.015)  # m: the clearance grid with omni (size, resolution): 3 x 3 rays, +-15 mm about the foot
STAIR_FAMILY_PREFIXES = ("stairs", "jstairs")
MODES = ("forward", "omni", "spin", "stand", "walk_turn")


@dataclass
class WalkerOptions(TerrainOptions):
    terrain: str = "narrow"                 # "narrow" (the walker's), "rough" (common.env's), "flat"
    fixed_command: tuple | None = None
    level: int | None = None
    narrow_risers: tuple | None = None      # the narrow flights' nominal riser range (m); None = (0.010, 0.035)
    boxes_share: float | None = None        # the box obstacles' share (narrow mix); None = the mix's own
    vx_max: float = 0.15                    # the forward draw's top (stair columns keep it; omni columns ramp it)
    wz_max: float = 0.6                     # the forward draw's yaw rate, U(+- this), on non-heading envs
    rel_standing: float = 0.1
    resample_s: tuple = (5.0, 10.0)
    push_mps: float = 0.05                  # mjlab's push (box +- this, every 3-6 s): only when no PushKick tier
    push_range: tuple | None = None         # PushKick tier 1: |dv| U(lo, hi) m/s, uniform horizontal direction
    push_interval: tuple | None = None      # tier 1: s between kicks, U(A, B), restarted at each reset
    topple_range: tuple | None = None       # PushKick tier 2
    topple_interval: tuple | None = None
    topple_start_step: int = 0              # env step from which tier 2 kicks (before: its timers redraw, no kick)
    push_fall_grace: float = 0.0            # s: > 0, the curriculum spares pushed episodes
    push_obs: bool = False                  # an observation group "push_t" (the time since the last kick) for
                                            # AnchorPPO's anchor-push mask; no network reads it
    heading: str = "loop"                   # "loop": wz from a heading loop on most envs (the only form)
    heading_kp: float = 2.0
    heading_turn: float = 1.0               # rad: new targets within +- this of the current heading
    rel_heading: float = 0.75
    heading_law: str = "p"                  # "p": mjlab's clip(kp err, +-wz_max); "pi": the robot's (heading.pi_step)
    heading_wz: float = 0.4                 # "pi": the loop's output limit (rad/s)
    alive: float = 0.0
    terminated: float = 0.0
    progress: float = 0.0                   # weight per second of progress_linear
    overspeed: float = 0.0                  # weight (< 0) of overspeed_omni (needs omni_mix)
    omni_promote_rel: float = 0.0           # > 0: an omni episode is promoted at min(promote_m, this x its commanded path)
    stumble: float = 0.0                    # weight per second per foot pushed mostly sideways
    clearance: str = "speed"                # "speed": foot_clearance_now; "apex": SwingApexClearance
    clearance_weight: float = -4.0
    clearance_target: float = 0.025
    foot_softness: tuple | None = None      # the feet's solref timeconst drawn per env (s)
    abduction: float = 0.0                  # weight per second of the abductions' squared distance from the stand
    leg_contact: float = -0.5
    stair_exit: bool = False
    omni_mix: str | None = None             # a key of OMNI_MIXES (None: forward commands only)
    omni_shares: tuple | None = None        # five shares overriding the mix key's (forward omni spin stand walk_turn)
    # the command range curriculum: each (start, end); start = end is no curriculum
    fwd_vx_range: tuple | None = None       # the forward draw's top on the omni columns (None: (vx_max, vx_max))
    omni_speed_max: tuple = (OMNI_SPEED[1], OMNI_SPEED[1])
    omni_wz: tuple = (OMNI_WZ, OMNI_WZ)
    spin_wz_max: tuple = (SPIN_WZ[1], SPIN_WZ[1])
    turn_wz_max: tuple = (TURN_WZ[1], TURN_WZ[1])
    cmd_ramp_iters: int = 0                 # iterations from start to end (0: the end values from the start)
    cmd_ramp_start_step: int = 0            # the env step the ramp starts at (a resumed run's first: iteration x 24)
    cmd_fast_share: float = 0.0             # a share of the omni columns' walks and spins drawn from the fast bands
    fast_speed: tuple = (0.15, 0.20)        # m/s: a fast forward / omni walk / walk-and-turn speed, U(band) (capped at
                                            # the range in force)
    fast_wz: tuple = (1.5, 2.0)             # rad/s: a fast spin's |wz|, U(band) (capped at the range in force)
    yaw_zero_share: float = 0.0             # a share of the non-held omni walks commanded with wz = 0 exactly
    yaw_hold_weight: float = 0.0            # weight per second of yaw_hold (tight yaw-rate tracking on straight walks)
    yaw_hold_std: float = 0.10              # rad/s: yaw_hold's kernel width
    spin_progress_weight: float = 0.0       # weight per second of spin_progress (partial yaw tracking pays, linear)
    yaw_sign_stats: bool = True             # log the yaw tracking per sign of the commanded turn (YawSignStats)
    attachment: Attachment | None = None


def mix_shares(opt: WalkerOptions) -> tuple | None:
    """The five shares (forward, omni walk, spin, standing, walk-and-turn), or None (no omni mix)."""
    if opt.omni_mix is None:
        return None
    s = tuple(float(x) for x in (opt.omni_shares if opt.omni_shares is not None else OMNI_MIXES[opt.omni_mix]))
    if len(s) == 4:
        s = s + (0.0,)
    if len(s) != 5 or min(s) < 0 or sum(s) <= 0:
        raise ValueError(f"omni shares: five values >= 0, not {s}")
    return s


def omni_columns(subs: dict) -> tuple:
    """The generator's columns the omni mix applies to: every family that is not a staircase."""
    from mjlab.terrains.primitive_terrains import (
        BoxInvertedPyramidStairsTerrainCfg,
        BoxPyramidStairsTerrainCfg,
    )
    stair = (wt.JitterPyramidStairsCfg, terrain_mod.StairGridCfg, BoxPyramidStairsTerrainCfg,
             BoxInvertedPyramidStairsTerrainCfg)
    return tuple(i for i, (k, c) in enumerate(subs.items())
                 if not isinstance(c, stair) and not k.startswith(STAIR_FAMILY_PREFIXES))


def ramp_fraction(step: int, start_step: int, ramp_iters: int, steps_per_iter: int = 24) -> float:
    """The range curriculum's position, 0 at ``start_step`` to 1 after ``ramp_iters`` iterations (1 when that is 0)."""
    if ramp_iters <= 0:
        return 1.0
    return min(1.0, max(0.0, (int(step) - int(start_step)) / float(ramp_iters * steps_per_iter)))


def lerp(r: tuple, f: float) -> float:
    return float(r[0]) + f * (float(r[1]) - float(r[0]))


class RelativeHeadingCommand(UniformVelocityCommand):
    """mjlab's heading command with each new target drawn relative to the current heading, applied at the next update.
    ``heading_law`` "pi": the heading envs' wz from the robot's PI loop. On ``narrow_cols`` every env is a heading env
    walking forward at U(``narrow_vx``), never standing, holding its spawn heading for the episode.

    ``omni_mix`` (five shares) on ``omni_cols``: each resample draws a mode per env (``omni_mode``: -1 off those columns,
    0 forward, 1 omni walk, 2 spin, 3 standing, 4 walk-and-turn), after the forward draw (whose draws it leaves as they
    were). The random draws for modes 0-3 come first, in a fixed order; walk-and-turn draws two more only when its
    share is > 0. ``env.walker_omni_ep`` marks the episodes with an omni walk, a spin or a walk-and-turn. The ranges follow
    the curriculum (``ranges_now``)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.omni_mode = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        if cfg.omni_mix is not None:
            env.walker_omni_ep = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._offset = torch.zeros(self.num_envs, device=self.device)
        self._pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._integ = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.fast_stats = {"fast_frac": 0.0, "yaw_zero_frac": 0.0}    # the last resample's (omni columns' walks/spins)

    def ranges_now(self) -> dict:
        c = self.cfg
        f = ramp_fraction(self._env.common_step_counter, c.ramp_start_step, c.ramp_iters)
        return {"frac": f, "fwd_vx_max": lerp(c.fwd_vx, f), "omni_speed_max": lerp(c.omni_speed_max, f),
                "omni_wz": lerp(c.omni_wz, f), "spin_wz_max": lerp(c.spin_wz_max, f),
                "turn_wz_max": lerp(c.turn_wz_max, f)}

    def _narrow(self, env_ids, cols=None):
        cols = self.cfg.narrow_cols if cols is None else cols
        t = getattr(self._env.scene, "terrain", None)
        if not cols or t is None or getattr(t, "terrain_types", None) is None:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        return torch.isin(t.terrain_types[env_ids], torch.tensor(cols, device=self.device))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._fresh[ids] = True
        if self.cfg.omni_mix is not None:
            self._env.walker_omni_ep[ids] = False
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        keep = self.heading_target[env_ids].clone()
        keep_i = self._integ[env_ids].clone()
        super()._resample_command(env_ids)
        if self.cfg.heading_command:
            self._offset[env_ids] = self.heading_target[env_ids]
            self._pending[env_ids] = True
            self._integ[env_ids] = 0.0
            nar = self._narrow(env_ids)
            if bool(nar.any()):
                ids = env_ids[nar]
                fresh = self._fresh[ids]
                self.is_heading_env[ids] = True
                self.is_standing_env[ids] = False
                self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.narrow_vx)
                self.vel_command_b[ids, 1] = 0.0
                self._offset[ids] = 0.0
                self._pending[ids] = fresh
                self.heading_target[ids] = torch.where(fresh, self.heading_target[ids], keep[nar])
                self._integ[ids] = torch.where(fresh, 0.0, keep_i[nar])
        if self.cfg.omni_mix is not None:
            self._omni(env_ids)
        self._fresh[env_ids] = False

    def _omni(self, env_ids):
        on = self._narrow(env_ids, self.cfg.omni_cols)
        self.omni_mode[env_ids] = -1
        if not bool(on.any()):
            return
        ids = env_ids[on]
        n, dev = len(ids), self.device
        r = self.ranges_now()
        mix = self.cfg.omni_mix
        p = torch.tensor(mix, device=dev).cumsum(0)
        mode = torch.bucketize(torch.rand(n, device=dev) * p[-1], p, right=True).clamp(max=len(mix) - 1)
        walk, spin, stand, turn = mode == 1, mode == 2, mode == 3, mode == 4
        speed = torch.empty(n, device=dev).uniform_(OMNI_SPEED[0], r["omni_speed_max"])
        ang = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
        held = (torch.rand(n, device=dev) < OMNI_HEADING_SHARE) & bool(self.cfg.heading_command)
        wz_walk = torch.empty(n, device=dev).uniform_(-r["omni_wz"], r["omni_wz"])
        sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        wz_spin = sign * torch.empty(n, device=dev).uniform_(SPIN_WZ[0], r["spin_wz_max"])
        if mix[4] > 0:      # walk-and-turn: the omni walk's speed and angle, |wz| U(TURN_WZ[0], max), random sign
            sign_t = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
            wz_turn = sign_t * torch.empty(n, device=dev).uniform_(TURN_WZ[0], r["turn_wz_max"])
        else:
            wz_turn = torch.zeros(n, device=dev)
        moving = walk | turn
        c = self.vel_command_b[ids]
        base_vx = float(self.cfg.ranges.lin_vel_x[1])
        fwd_k = r["fwd_vx_max"] / base_vx if base_vx > 0 else 1.0
        fwd_vx = c[:, 0] * fwd_k                                                    # forward: U(0, fwd_vx_max)
        # the fast bands and the zero-yaw walks draw random numbers only when on (at 0 the stream is unchanged)
        if self.cfg.fast_share > 0:
            fast = torch.rand(n, device=dev) < self.cfg.fast_share
            lo, hi = self.cfg.fast_speed
            v_fast = torch.empty(n, device=dev).uniform_(lo, hi)
            w_lo, w_hi = self.cfg.fast_wz
            w_fast = torch.empty(n, device=dev).uniform_(w_lo, w_hi)
            speed = torch.where(fast, v_fast.clamp(max=r["omni_speed_max"]), speed)
            fwd_vx = torch.where(fast, v_fast.clamp(max=r["fwd_vx_max"]), fwd_vx)
            wz_spin = torch.where(fast, sign * w_fast.clamp(max=r["spin_wz_max"]), wz_spin)
            drawn = mode != 3
            self.fast_stats["fast_frac"] = float((fast & drawn).sum()) / max(int(drawn.sum()), 1)
        if self.cfg.yaw_zero_share > 0:
            zero = (torch.rand(n, device=dev) < self.cfg.yaw_zero_share) & ~held
            wz_walk = torch.where(zero, 0.0, wz_walk)
            self.fast_stats["yaw_zero_frac"] = float((zero & walk).sum()) / max(int(walk.sum()), 1)
        c[:, 0] = torch.where(moving, speed * torch.cos(ang),
                              torch.where(spin | stand, 0.0, fwd_vx))
        c[:, 1] = torch.where(moving, speed * torch.sin(ang), torch.where(spin | stand, 0.0, c[:, 1]))
        c[:, 2] = torch.where(walk, wz_walk, torch.where(spin, wz_spin, torch.where(
            turn, wz_turn, torch.where(stand, 0.0, c[:, 2]))))
        self.vel_command_b[ids] = c
        self.vel_command_w[ids] = c
        self.is_standing_env[ids] = stand
        h = self.is_heading_env[ids]
        self.is_heading_env[ids] = torch.where(mode == 0, h, walk & held)
        if self.cfg.heading_command:
            self._offset[ids] = torch.where(walk & held, 0.0, self._offset[ids])
        self.omni_mode[ids] = mode
        self._env.walker_omni_ep[ids] |= walk | spin | turn

    def _update_command(self):
        if self.cfg.heading_command and bool(self._pending.any()):
            from mjlab.utils.lab_api.math import wrap_to_pi
            ids = self._pending.nonzero(as_tuple=False).flatten()
            self.heading_target[ids] = wrap_to_pi(self.robot.data.heading_w[ids] + self._offset[ids])
            self._pending[ids] = False
        if self.cfg.heading_command and self.cfg.heading_law == "pi":
            from mjlab.utils.lab_api.math import wrap_to_pi

            from mp4_rl.common.heading import pi_step
            err = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            wz, integ = pi_step(err, self._integ, self._env.step_dt, wz_max=self.cfg.heading_wz)
            h = self.is_heading_env
            self._integ = torch.where(h, integ, self._integ)
            super()._update_command()
            self.heading_error = err
            self.vel_command_b[:, 2] = torch.where(h & ~self.is_standing_env, wz, self.vel_command_b[:, 2])
            return
        super()._update_command()


@dataclass(kw_only=True)
class RelativeHeadingCommandCfg(UniformVelocityCommandCfg):
    heading_law: str = "p"
    heading_wz: float = 0.4
    narrow_cols: tuple = ()
    narrow_vx: tuple = (0.06, 0.12)
    omni_mix: tuple | None = None
    omni_cols: tuple = ()
    fwd_vx: tuple = (0.15, 0.15)
    omni_speed_max: tuple = (OMNI_SPEED[1], OMNI_SPEED[1])
    omni_wz: tuple = (OMNI_WZ, OMNI_WZ)
    spin_wz_max: tuple = (SPIN_WZ[1], SPIN_WZ[1])
    turn_wz_max: tuple = (TURN_WZ[1], TURN_WZ[1])
    ramp_iters: int = 0
    ramp_start_step: int = 0
    fast_share: float = 0.0
    fast_speed: tuple = (0.15, 0.20)
    fast_wz: tuple = (1.5, 2.0)
    yaw_zero_share: float = 0.0

    def build(self, env):
        return RelativeHeadingCommand(self, env)


def command_range(env, env_ids, command_name: str = "twist"):
    """The command ranges in force (``RelativeHeadingCommand.ranges_now``), logged as ``Curriculum/command_range/<k>``."""
    t = env.command_manager.get_term(command_name)
    if not hasattr(t, "ranges_now"):
        return None
    out = {k: torch.tensor(v) for k, v in t.ranges_now().items()}
    if t.cfg.fast_share > 0 or t.cfg.yaw_zero_share > 0:
        out.update({k: torch.tensor(v) for k, v in t.fast_stats.items()})
    return out


# ---------------------------------------------------------------------------------------------------- stairs
STAIR_EXIT_MARGIN = 0.08
STAIR_STAYED_M = 0.30
STAIR_FOOT_PAST = 0.01


def _narrow_tables(env, subs) -> None:
    if hasattr(env, "_walker_narrow"):
        return
    rows = env.scene.terrain.terrain_origins.shape[0]
    tab = torch.zeros(rows, len(subs), 3, device=env.device)
    nar = torch.zeros(len(subs), dtype=torch.bool, device=env.device)
    for col, cfg in enumerate(subs):
        if isinstance(cfg, wt.NarrowFlightCfg):
            f = cfg.flights[-rows:]
            tab[:, col] = torch.tensor([[x["x_first"], x["x_last"], x["width"] / 2] for x in f], device=env.device)
            nar[col] = True
    env._walker_flight, env._walker_narrow = tab, nar


def stair_exit(env, subs: tuple, tile: float):
    """True on a jittered-stair tile once the base is off the stairs (Chebyshev distance from the spawn past the
    outermost riser + ``STAIR_EXIT_MARGIN``), on a narrow flight once all four feet are on its landing: a success."""
    t = env.scene.terrain
    n = env.num_envs
    if not hasattr(env, "_walker_extent"):
        rows = t.terrain_origins.shape[0]
        ext = torch.full((rows, len(subs)), float("inf"), device=env.device)
        for col, cfg in enumerate(subs):
            if isinstance(cfg, wt.JitterPyramidStairsCfg):
                e = torch.tensor(cfg.extents[-rows:], device=env.device)
                ext[:, col] = torch.clamp(e + STAIR_EXIT_MARGIN, max=tile / 2 - 0.02)
        env._walker_extent = ext
        _narrow_tables(env, subs)
        env._walker_stairs = torch.isfinite(ext[0]) | env._walker_narrow
        env.walker_cheb_max = torch.zeros(n, device=env.device)
        env.walker_stair_exit = torch.zeros(n, dtype=torch.bool, device=env.device)
    p = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    cheb = p.abs().max(-1).values
    new = env.episode_length_buf <= 1
    env.walker_cheb_max = torch.where(new, cheb, torch.maximum(env.walker_cheb_max, cheb))
    env.walker_stair_exit = cheb >= env._walker_extent[t.terrain_levels, t.terrain_types]
    if bool(env._walker_narrow.any()):
        on = env._walker_narrow[t.terrain_types]
        rear = (env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 0]
                - env.scene.env_origins[:, :1]).min(-1).values
        landed = rear > env._walker_flight[t.terrain_levels, t.terrain_types, 1] + STAIR_FOOT_PAST
        env.walker_stair_exit = torch.where(on, landed, env.walker_stair_exit)
    return env.walker_stair_exit


def side_exit(env, subs: tuple, foot_clear: float):
    """True on a narrow flight once a stance foot past the first riser, or the base, is beyond the flight's half-width."""
    _narrow_tables(env, subs)
    if not bool(env._walker_narrow.any()):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    t = env.scene.terrain
    on = env._walker_narrow[t.terrain_types]
    tab = env._walker_flight[t.terrain_levels, t.terrain_types]
    o = env.scene.env_origins
    fp = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), :2] - o[:, None, :2]
    stance = (fp[..., 0] > tab[:, :1] + STAIR_FOOT_PAST) & (foot_heights_centre(env) <= foot_clear + 0.002)
    off = (stance & (fp[..., 1].abs() > tab[:, 2:3])).any(-1)
    base_y = (env.scene["robot"].data.root_link_pos_w[:, 1] - o[:, 1]).abs()
    return on & (off | (base_y > tab[:, 2]))


def reset_base_narrow(env, env_ids, pose_range: dict, narrow_pose_range: dict, narrow_cols: tuple):
    """The base reset: ``pose_range`` everywhere, then ``narrow_pose_range`` on the narrow flights (the start jig)."""
    vel_mdp.reset_root_state_uniform(env, env_ids, pose_range=pose_range, velocity_range={})
    t = env.scene.terrain
    if env_ids is None or not narrow_cols or t is None:
        return
    nar = torch.isin(t.terrain_types[env_ids], torch.tensor(narrow_cols, device=env.device))
    if bool(nar.any()):
        vel_mdp.reset_root_state_uniform(env, env_ids[nar], pose_range=narrow_pose_range, velocity_range={})


# ---------------------------------------------------------------------------------------------------- curriculum
class PathTracker(ManagerTermBase):
    """The progress walked (forward: the positive body-frame forward speed; omni: along the commanded direction, 0 below
    ``OMNI_MIN_CMD``) and the distance commanded, integrated per episode for the terrain curriculum. Returns 0."""

    def __init__(self, cfg, env):
        super().__init__(env)
        env.walker_path = torch.zeros(env.num_envs, device=env.device)
        env.walker_cmd_path = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._env.walker_path[ids] = 0
        self._env.walker_cmd_path[ids] = 0

    def __call__(self, env, command_name: str = "twist", omni: bool = False):
        cmd = env.command_manager.get_command(command_name)[:, :2]
        c = cmd.norm(dim=-1)
        if omni:
            v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
            along = ((v * cmd).sum(-1) / c.clamp(min=OMNI_MIN_CMD)).clamp(min=0.0)
            vx = torch.where(c >= OMNI_MIN_CMD, along, torch.zeros_like(along))
        else:
            vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
        env.walker_path += vx * env.step_dt
        env.walker_cmd_path += (torch.where(c >= OMNI_MIN_CMD, c, torch.zeros_like(c)) if omni else c) * env.step_dt
        return torch.zeros_like(vx)


def push_grace(env, env_ids, fell, up, down, grace_s: float):
    p = getattr(env, "mp4_push", None)
    if p is None:
        return up, down
    pushed = p.pushed[env_ids]
    grace = fell & pushed & (p.t_since[env_ids] + env.step_dt <= grace_s)
    keep = ~pushed | (fell & ~grace)
    p.note_curriculum(int((down & pushed & ~fell).sum()), int((down & grace).sum()))
    return up & ~grace, down & keep


def terrain_levels_path(env, env_ids, promote_m: float, min_cmd_m: float = 0.3, omni_rel: float = 0.0,
                        grace_s: float = 0.0):
    """Up a level when the episode walked at least ``promote_m`` and did not fall; down when it walked under half the
    commanded path or fell. Omni episodes (``env.walker_omni_ep``) are promoted at min(promote_m, ``omni_rel`` x the
    commanded path) and demoted only on a fall; stair tiles (``stair_exit``) up only off the stairs."""
    t = env.scene.terrain
    if t is None or getattr(t, "terrain_levels", None) is None or not hasattr(env, "walker_path"):
        return None
    path, cmd = env.walker_path[env_ids], env.walker_cmd_path[env_ids]
    fell = env.termination_manager.terminated[env_ids]
    judged = cmd >= min_cmd_m
    up = judged & ~fell & (path >= promote_m)
    down = judged & ~up & (fell | (path < 0.5 * cmd))
    omni = getattr(env, "walker_omni_ep", None)
    if omni is not None:
        if omni_rel > 0:
            up_rel = judged & ~fell & (path >= torch.clamp(omni_rel * cmd, max=promote_m))
            up = torch.where(omni[env_ids], up_rel, up)
        down = torch.where(omni[env_ids], ~up & fell, down)
    if hasattr(env, "walker_stair_exit"):
        st = env._walker_stairs[t.terrain_types[env_ids]]
        out = env.walker_stair_exit[env_ids]
        s_up = ~fell & out
        s_down = ~out & (fell | (judged & (env.walker_cheb_max[env_ids] < STAIR_STAYED_M)))
        up, down = torch.where(st, s_up, up), torch.where(st, s_down, down)
    if grace_s > 0:
        up, down = push_grace(env, env_ids, fell, up, down, grace_s)
    t.update_env_origins(env_ids, up, down)
    return torch.mean(t.terrain_levels.float())


def family_levels(env, env_ids, names: tuple):
    t = env.scene.terrain
    if t is None or getattr(t, "terrain_levels", None) is None:
        return None
    lv, ty = t.terrain_levels.float(), t.terrain_types
    return {n: lv[ty == i].mean() for i, n in enumerate(names) if bool((ty == i).any())}


# ---------------------------------------------------------------------------------------------------- rewards
def track_lin_vel_xy_exp(env, std: float, command_name: str = "twist"):
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-((v - cmd) ** 2).sum(-1) / std ** 2)


def track_ang_vel_z_exp(env, std: float, command_name: str = "twist"):
    w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-((w - cmd) ** 2) / std ** 2)


def yaw_hold(env, std: float = 0.10, command_name: str = "twist", wz_eps: float = 0.05, min_speed: float = 0.02):
    """Heading hold: exp(-((w - wz_cmd) / std)^2) on straight walks (|v_cmd| > ``min_speed``, |wz_cmd| <= ``wz_eps``),
    0 elsewhere. ``track_ang_vel``'s kernel (std 0.25 rad/s) pays a 0.13 rad/s yaw drift (90 deg in 12 s) 76 % of its
    reward; this term (std 0.10) pays it 18 %, so a slow drift on a straight walk costs."""
    cmd = env.command_manager.get_command(command_name)
    w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    on = (cmd[:, :2].norm(dim=-1) > min_speed) & (cmd[:, 2].abs() <= wz_eps)
    return torch.exp(-((w - cmd[:, 2]) / std) ** 2) * on.float()


SPIN_PROGRESS_MIN_WZ = 0.3     # rad/s: spin_progress and the per-sign yaw log act above this |wz_cmd|


def spin_progress(env, command_name: str = "twist", min_wz: float = SPIN_PROGRESS_MIN_WZ):
    """clip(w * wz_cmd / wz_cmd^2, 0, 1) = clip(w / wz_cmd, 0, 1) for spins and walk-and-turns (``omni_mode`` 2, 4) with
    |wz_cmd| > ``min_wz``: linear in the yaw rate achieved in the commanded direction, so a partial turn pays and
    freezing or turning the wrong way pays 0. ``track_ang_vel``'s exp kernel (std 0.25) pays about nothing for a spin
    more than ~0.6 rad/s short, which lets a frozen pose beat a partial spin."""
    term = env.command_manager.get_term(command_name)
    wz = env.command_manager.get_command(command_name)[:, 2]
    w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    mode = getattr(term, "omni_mode", None)
    turning = ((mode == 2) | (mode == 4)) if mode is not None else torch.zeros_like(wz, dtype=torch.bool)
    on = turning & (wz.abs() > min_wz)
    return (w * wz / (wz * wz).clamp(min=1e-6)).clamp(0.0, 1.0) * on.float()


class YawSignStats(ManagerTermBase):
    """Accumulates, over each rollout block of ``block_steps`` env steps, the yaw tracking of the samples commanded to
    turn clockwise (wz_cmd < -``min_wz``) and anticlockwise (> ``min_wz``): mean |w - wz_cmd|, mean w / wz_cmd (the
    achieved over the commanded rate) and the samples' share. Returns 0 (a reward term only to run every step); the
    last closed block is ``env.yaw_sign_stats``, logged by ``yaw_sign_log`` as ``Curriculum/yaw_sign/<key>``."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._block = 24
        self._b = None
        self._acc = self._zero()
        env.yaw_sign_stats = {}

    @staticmethod
    def _zero():
        return {k: 0.0 for k in ("n_cw", "err_cw", "ratio_cw", "n_ccw", "err_ccw", "ratio_ccw", "n")}

    def __call__(self, env, command_name: str = "twist", min_wz: float = SPIN_PROGRESS_MIN_WZ):
        b = int(env.common_step_counter) // self._block
        if self._b is None:
            self._b = b
        if b != self._b:
            a = self._acc
            out = {}
            for side in ("cw", "ccw"):
                n = a[f"n_{side}"]
                out[f"err_{side}"] = a[f"err_{side}"] / n if n else 0.0
                out[f"ratio_{side}"] = a[f"ratio_{side}"] / n if n else 0.0
                out[f"share_{side}"] = n / a["n"] if a["n"] else 0.0
            env.yaw_sign_stats = out
            self._acc, self._b = self._zero(), b
        wz = env.command_manager.get_command(command_name)[:, 2]
        w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
        self._acc["n"] += wz.numel()
        for side, m in (("cw", wz < -min_wz), ("ccw", wz > min_wz)):
            k = int(m.sum())
            if k:
                self._acc[f"n_{side}"] += k
                self._acc[f"err_{side}"] += float((w[m] - wz[m]).abs().sum())
                self._acc[f"ratio_{side}"] += float((w[m] / wz[m]).sum())
        return torch.zeros_like(wz)


YAW_SIGN_KEYS = tuple(f"{k}_{side}" for side in ("cw", "ccw") for k in ("err", "ratio", "share"))


def yaw_sign_log(env, env_ids):
    """``env.yaw_sign_stats``; every key from the first call (zeros before the first closed block): rsl_rl's logger
    prints only the keys of a rollout's first step's log."""
    s = getattr(env, "yaw_sign_stats", None) or {}
    return {k: torch.tensor(float(s.get(k, 0.0))) for k in YAW_SIGN_KEYS}


def lin_vel_z_l2(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2] ** 2


def ang_vel_xy_l2(env):
    return (env.scene["robot"].data.root_link_ang_vel_b[:, :2] ** 2).sum(-1)


def joint_acc_l2(env):
    return (env.scene["robot"].data.joint_acc ** 2).sum(-1)


def stance_foot_slip_l2(env, sensor_name: str = "feet_contact", threshold: float = 1.0):
    s = env.scene[sensor_name].data
    contact = (s.force.norm(dim=-1) > threshold).float()
    v = env.scene["robot"].data.site_lin_vel_w[:, _foot_site_ids(env), :2]
    return (contact * (v ** 2).sum(-1)).sum(-1)


def stand_still_l2(env, command_name: str = "twist", deadband: float = 1e-3):
    """With no command, the joints' squared distance from the stand."""
    cmd = env.command_manager.get_command(command_name)
    idle = (cmd.abs().amax(dim=-1) < deadband).float()
    return idle * (mdp.servo_action(env).true_rel() ** 2).sum(-1)


def progress_linear(env, command_name: str = "twist", min_cmd: float = 0.02):
    """Body speed along the commanded direction over the commanded speed, clipped to [0, 1]; 0 below ``min_cmd``."""
    c = env.command_manager.get_command(command_name)[:, :2]
    speed = c.norm(dim=-1)
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    along = (v * c).sum(-1) / speed.clamp(min=min_cmd)
    return (along / speed.clamp(min=min_cmd)).clamp(0.0, 1.0) * (speed >= min_cmd).float()


def overspeed_omni(env, command_name: str = "twist", min_cmd: float = 0.02):
    """The speed along the command above it, over it, on the omni columns' forward, omni and walk-and-turn walks."""
    term = env.command_manager.get_term(command_name)
    c = env.command_manager.get_command(command_name)[:, :2]
    speed = c.norm(dim=-1)
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    along = (v * c).sum(-1) / speed.clamp(min=min_cmd)
    over = (along / speed.clamp(min=min_cmd) - 1.0).clamp(min=0.0)
    mode = getattr(term, "omni_mode", None)
    walk = ((mode == 0) | (mode == 1) | (mode == 4)) if mode is not None else torch.zeros_like(speed, dtype=torch.bool)
    return over * (walk & (speed >= min_cmd)).float()


def feet_stumble(env, ratio: float = 1.5, sensor_name: str = "feet_contact"):
    """Feet in contact with a mostly horizontal net force, |F_xy| > ratio |F_z|."""
    d = env.scene[sensor_name].data
    n = d.found.shape[1]
    f = d.force.view(d.force.shape[0], n, 3)
    hit = (d.found > 0) & (f[..., :2].norm(dim=-1) > ratio * f[..., 2].abs())
    return hit.float().sum(-1)


@dataclass
class AheadPatternCfg(GridPatternCfg):
    """A grid of rays shifted by ``offset`` in the frame (yaw-aligned), started high."""
    offset: tuple = (0.015, 0.0, 0.05)

    def generate_rays(self, mj_model, device):
        off, dirs = super().generate_rays(mj_model, device)
        return off + torch.tensor(self.offset, device=device, dtype=off.dtype), dirs


def ground_ahead(env, ahead: str = "foot_ahead", under: str = "foot_height") -> torch.Tensor:
    da, du = env.scene[ahead].data, env.scene[under].data
    B, F = du.heights.shape[0], du.heights.shape[1]
    za = torch.where(da.distances.view(B, F, -1) >= 0, da.hit_pos_w[..., 2].view(B, F, -1),
                     torch.full_like(da.distances.view(B, F, -1), -1.0e3)).amax(-1)
    zu = torch.where(du.distances.view(B, F, -1) >= 0, du.hit_pos_w[..., 2].view(B, F, -1),
                     torch.full_like(du.distances.view(B, F, -1), -1.0e3)).amax(-1)
    return torch.maximum(za, zu)


class SwingApexClearance(ManagerTermBase):
    """At each touchdown after at least ``min_air`` s of swing, how far the swing's highest clearance (the foot's bottom
    over the highest terrain under it or ahead of it) stayed below ``target``; metres per touchdown / dt. Only while
    commanded to move. ``foot_clear``: a foot site's height over flat ground at contact (the legs')."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._apex = torch.zeros(env.num_envs, 4, device=env.device)
        self._air = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self._air_s = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._apex[ids] = 0.0
        self._air[ids] = False
        self._air_s[ids] = 0.0

    def __call__(self, env, target: float, foot_clear: float, min_air: float = 0.06, command_name: str = "twist",
                 command_threshold: float = 0.02):
        found = env.scene["feet_contact"].data.found > 0
        z = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 2]
        clear = z - foot_clear - ground_ahead(env)
        air = ~found
        self._apex = torch.where(air & self._air, torch.maximum(self._apex, clear), torch.where(air, clear, self._apex))
        touchdown = self._air & found & (self._air_s >= min_air - 1e-6)
        deficit = (target - self._apex).clamp(0.0, target) * touchdown.float()
        self._air_s = torch.where(air, self._air_s + env.step_dt, torch.zeros_like(self._air_s))
        self._air = air
        cmd = env.command_manager.get_command(command_name)
        active = (cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()) > command_threshold
        return deficit.sum(-1) * active.float() / env.step_dt


def foot_heights_now(env, sensor_name: str = "foot_height") -> torch.Tensor:
    d = env.scene[sensor_name].data
    B, F = d.heights.shape[0], d.heights.shape[1]
    hit_z = d.hit_pos_w[..., 2].view(B, F, -1)
    hit = (d.distances.view(B, F, -1) >= 0)
    ground = torch.where(hit, hit_z, torch.full_like(hit_z, -1.0e3)).amax(-1)
    site = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 2]
    return torch.where(hit.any(-1), site - ground, torch.zeros_like(site))


def foot_clearance_now(env, target_height: float, command_name: str = "twist", command_threshold: float = 0.02):
    v = env.scene["robot"].data.site_lin_vel_w[:, _foot_site_ids(env), :2].norm(dim=-1)
    cost = ((foot_heights_now(env) - target_height).abs() * v).sum(-1)
    cmd = env.command_manager.get_command(command_name)
    active = (cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()) > command_threshold
    return cost * active.float()


@requires_model_fields("geom_solref")
def foot_softness(env, env_ids, ranges, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The feet's contact timeconst (solref[0]) drawn per env, the same for all four."""
    from mjlab.envs.mdp.dr._core import _randomize_model_field
    _randomize_model_field(env, env_ids, "geom_solref", entity_type="geom", ranges=ranges, operation="abs",
                           asset_cfg=asset_cfg, axes=[0], shared_random=True, default_axes=[0], valid_axes=[0, 1])


def terrain_normal(env, sensor_name: str = "body_scan") -> torch.Tensor:
    d = env.scene[sensor_name].data
    n = d.normals_w
    hit = (d.distances >= 0) & torch.isfinite(n).all(-1)
    n = torch.where(hit.unsqueeze(-1), n, torch.zeros_like(n)).sum(1)
    up = torch.tensor([0.0, 0.0, 1.0], device=n.device).expand_as(n)
    norm = n.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, n / norm.clamp(min=1e-6), up)


def upright_terrain_l2(env, sensor_name: str = "body_scan"):
    from mjlab.utils.lab_api.math import quat_apply_inverse
    nb = quat_apply_inverse(env.scene["robot"].data.root_link_quat_w, terrain_normal(env, sensor_name))
    return (nb[:, :2] ** 2).sum(-1)


def abduction_deviation_l2(env):
    return (mdp.servo_action(env).true_rel()[:, list(ABDUCTION)] ** 2).sum(-1)


def leg_contacts(env, sensor_name: str = "leg_contact"):
    return (env.scene[sensor_name].data.found > 0).float().sum(-1)


BODY_SCAN_STAND_Z = 0.0725      # m: the body scan's reference on SKU2 (another robot: + robot_z_shift)


def obs_body_scan(env, sensor_name: str = "body_scan", clip: float = 0.15, stand_z: float = BODY_SCAN_STAND_Z):
    """Heights of the base above the terrain at the scan points, minus the stand height (72.5 mm on SKU2), clipped."""
    h = envs_mdp.height_scan(env, sensor_name)
    return (h - stand_z).clamp(-clip, clip)


def obs_foot_heights(env, foot_clear: float, sensor_name: str = "foot_height"):
    return (env.scene[sensor_name].data.heights - foot_clear).clamp(-0.05, 0.10)


# ---------------------------------------------------------------------------------------------------- kicks
NO_PUSH_S = 1000.0


class PushKick(ManagerTermBase):
    """Kicks as base velocity changes: each tier ``(lo, hi, a, b)`` adds |dv| ~ U(lo, hi) in a uniformly random horizontal
    direction to the base's velocity every U(a, b) s per env (timers drawn at each reset and after each kick). Per env
    ``t_since``, ``last_mag``, ``pushed``; ``env.mp4_push`` is this term; ``stats`` the last rollout's kicks per env-second,
    mean |dv| per tier and the curriculum's spared demotions per kick. Use with mode="step"."""

    def __init__(self, cfg, env):
        super().__init__(env)
        p = cfg.params
        self._asset = env.scene[p["asset_cfg"].name]
        self._tiers = tuple(tuple(float(x) for x in t) for t in p["tiers"])
        self._grace = float(p.get("grace_s", 2.0))
        self._block = int(p.get("block_steps", 24))
        self._starts = tuple(int(x) for x in (p.get("start_steps") or (0,) * len(self._tiers)))
        assert len(self._starts) == len(self._tiers), (self._starts, self._tiers)
        n, dev = env.num_envs, env.device
        self.time_left = torch.zeros(len(self._tiers), n, device=dev)
        self.t_since = torch.full((n,), NO_PUSH_S, device=dev)
        self.last_mag = torch.zeros(n, device=dev)
        self.pushed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._acc = self._zero_acc()
        self._acc_block = None
        self.stats = {}
        env.mp4_push = self

    def _zero_acc(self):
        k = len(self._tiers)
        return {"kicks": [0] * k, "dv": [0.0] * k, "env_steps": 0, "grace_falls": 0, "spared_shortfall": 0,
                "spared_fall": 0}

    def _roll(self, env) -> None:
        b = int(env.common_step_counter) // self._block
        if self._acc_block is None:
            self._acc_block = b
        if b == self._acc_block:
            return
        a, dt = self._acc, env.step_dt
        kicks = sum(a["kicks"])
        out = {}
        for i in range(len(self._tiers)):
            out[f"kicks_per_env_s_t{i + 1}"] = a["kicks"][i] / max(a["env_steps"] * dt, 1e-9)
            out[f"mean_dv_t{i + 1}"] = a["dv"][i] / a["kicks"][i] if a["kicks"][i] else 0.0
        for k in ("grace_falls", "spared_shortfall", "spared_fall"):
            out[f"{k}_per_kick"] = a[k] / kicks if kicks else 0.0
        out["kicks"] = float(kicks)
        self.stats = out
        self._acc, self._acc_block = self._zero_acc(), b

    def note_curriculum(self, spared_shortfall: int, spared_fall: int) -> None:
        self._acc["spared_shortfall"] += spared_shortfall
        self._acc["spared_fall"] += spared_fall

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            ids, n = slice(None), self._env.num_envs
        else:
            ids, n = env_ids, len(env_ids)
            tm = getattr(self._env, "termination_manager", None)
            if tm is not None and n:
                fell = tm.terminated[env_ids]
                near = self.pushed[env_ids] & (self.t_since[env_ids] + self._env.step_dt <= self._grace)
                self._acc["grace_falls"] += int((fell & near).sum())
        for k, (_, _, a, b) in enumerate(self._tiers):
            self.time_left[k, ids] = torch.empty(n, device=self._env.device).uniform_(a, b)
        self.t_since[ids] = NO_PUSH_S
        self.last_mag[ids] = 0.0
        self.pushed[ids] = False

    def __call__(self, env, env_ids, tiers, asset_cfg, grace_s: float = 2.0, block_steps: int = 24,
                 start_steps: tuple | None = None) -> None:
        self._roll(env)
        dt = env.step_dt
        live = env.episode_length_buf > 0
        self.t_since = torch.where(live, (self.t_since + dt).clamp(max=NO_PUSH_S), self.t_since)
        self.time_left -= dt * live.float()
        self._acc["env_steps"] += env.num_envs
        dv = torch.zeros(env.num_envs, 2, device=env.device)
        hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for k, (lo, hi, a, b) in enumerate(self._tiers):
            due = live & (self.time_left[k] <= 0)
            m = int(due.sum())
            if not m:
                continue
            if int(env.common_step_counter) < self._starts[k]:     # a tier not started yet: redraw, no kick
                self.time_left[k, due] = torch.empty(m, device=env.device).uniform_(a, b)
                continue
            mag = torch.empty(m, device=env.device).uniform_(lo, hi)
            ang = torch.empty(m, device=env.device).uniform_(-math.pi, math.pi)
            dv[due] += torch.stack((mag * torch.cos(ang), mag * torch.sin(ang)), dim=-1)
            self.time_left[k, due] = torch.empty(m, device=env.device).uniform_(a, b)
            hit |= due
            self._acc["kicks"][k] += m
            self._acc["dv"][k] += float(mag.sum())
        if not bool(hit.any()):
            return
        ids = hit.nonzero(as_tuple=False).squeeze(-1)
        vel = self._asset.data.root_link_vel_w[ids].clone()
        vel[:, :2] += dv[ids]
        self._asset.write_root_link_velocity_to_sim(vel, env_ids=ids)
        self.t_since[ids] = 0.0
        self.last_mag[ids] = dv[ids].norm(dim=-1)
        self.pushed[ids] = True


def obs_push_t(env):
    """The time since the last kick (s; ``NO_PUSH_S`` before the episode's first): the ``push_t`` group."""
    return env.mp4_push.t_since[:, None].clone()


def push_stats(env, env_ids):
    p = getattr(env, "mp4_push", None)
    return dict(p.stats) if p is not None and p.stats else None


def push_start_steps(opt: WalkerOptions) -> tuple:
    """Each ``push_tiers`` tier's first env step (tier 1 from the start, tier 2 from ``topple_start_step``)."""
    out = []
    if opt.push_range is not None:
        out.append(0)
    if opt.topple_range is not None:
        out.append(int(opt.topple_start_step))
    return tuple(out)


def push_tiers(opt: WalkerOptions) -> tuple:
    out = []
    for r, i, name in ((opt.push_range, opt.push_interval, "push"), (opt.topple_range, opt.topple_interval, "topple")):
        if r is None and i is None:
            continue
        if r is None or i is None:
            raise ValueError(f"{name}_range and {name}_interval go together")
        lo, hi = (float(x) for x in r)
        a, b = (float(x) for x in i)
        if not (0 <= lo <= hi and 0 < a <= b):
            raise ValueError(f"{name}: {r}, {i}")
        out.append((lo, hi, a, b))
    return tuple(out)


# ---------------------------------------------------------------------------------------------------- the env
SENSORS = ("feet_contact", "body_contact", "body_scan", "foot_height", "leg_contact")   # mp4_sim.envs.scene


def sub_terrains(opt: WalkerOptions) -> dict:
    if opt.boxes_share is not None and opt.terrain != "narrow":
        raise ValueError(f"boxes_share is the narrow mix's option, not {opt.terrain!r}'s")
    if opt.terrain == "narrow":
        return wt.narrow_sub_terrains(opt.tile, riser_range=opt.narrow_risers, boxes_share=opt.boxes_share)
    if opt.terrain == "rough":
        return terrain_mod.rough_sub_terrains(opt.tile)
    if opt.terrain == "flat":
        from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
        return {"flat": BoxFlatTerrainCfg(size=(opt.tile, opt.tile))}
    raise ValueError(f"unknown walker terrain {opt.terrain!r}")


def make_walker_env_cfg(opt: WalkerOptions | None = None):
    opt = opt or WalkerOptions()
    if opt.heading != "loop":
        raise ValueError("the walker trainer ports the heading loop only (heading='loop')")
    if opt.fixed_command is not None or opt.level is not None:
        raise ValueError("fixed_command / level are the evaluation env's (common.env.make_terrain_env_cfg)")
    subs = sub_terrains(opt)
    rows = 1 if opt.terrain == "flat" else opt.rows
    cfg, sc = env_and_scene(opt, sub_terrains=subs, rows=rows, sensors=SENSORS, attachment=opt.attachment)
    half_x, half_y = sc.half_x, sc.half_y
    fc = foot_r(opt)

    foot_height = next(s for s in cfg.scene.sensors if s.name == "foot_height")
    if opt.clearance == "apex":
        ahead = (AheadPatternCfg(size=(OMNI_CLEAR_GRID[0],) * 2, resolution=OMNI_CLEAR_GRID[1], offset=(0.0, 0.0, 0.05))
                 if opt.omni_mix is not None else AheadPatternCfg(size=(0.03, 0.01), resolution=0.01))
        cfg.scene.sensors += (RayCastSensorCfg(name="foot_ahead", frame=foot_height.frame, pattern=ahead,
                                               ray_alignment="yaw", max_distance=0.3, exclude_parent_body=True,
                                               include_geom_groups=(0,)),)

    # observations: the critic sees the terrain; the anchor and cmd groups ride in the rollout storage
    shift = robot_z_shift(legs_of(opt))     # 0.0 on SKU2: the scan's reference moves with another robot's stand
    cfg.observations["critic"].terms["body_scan"] = ObservationTermCfg(
        func=obs_body_scan, params={"stand_z": BODY_SCAN_STAND_Z + shift} if shift != 0.0 else {})
    cfg.observations["critic"].terms["foot_heights"] = ObservationTermCfg(func=obs_foot_heights,
                                                                          params={"foot_clear": fc})
    if opt.history:
        cfg.observations["actor"].history_length = opt.history
    clean = {k: ObservationTermCfg(func=v.func, params={**(v.params or {}), **({"noisy": False}
                                                                                if "noisy" in (v.params or {}) else {})})
             for k, v in cfg.observations["actor"].terms.items()}
    cfg.observations["anchor"] = ObservationGroupCfg(terms=clean, concatenate_terms=True, enable_corruption=False)
    cfg.observations["cmd"] = ObservationGroupCfg(terms={"command": ObservationTermCfg(func=mdp.obs_command)},
                                                  concatenate_terms=True, enable_corruption=False)

    if opt.push_obs:
        if not (push_tiers(opt) and opt.randomise):
            raise ValueError("push_obs needs a PushKick tier (and randomise)")
        cfg.observations["push_t"] = ObservationGroupCfg(terms={"t": ObservationTermCfg(func=obs_push_t)},
                                                         concatenate_terms=True, enable_corruption=False)

    # commands
    narrow_cols = tuple(i for i, c in enumerate(subs.values()) if isinstance(c, wt.NarrowFlightCfg))
    shares = mix_shares(opt)
    if opt.heading_law not in ("p", "pi"):
        raise ValueError(opt.heading_law)
    if narrow_cols and not opt.stair_exit:
        raise ValueError("narrow flights need stair_exit")
    fwd = tuple(opt.fwd_vx_range) if opt.fwd_vx_range is not None else (opt.vx_max, opt.vx_max)
    cfg.commands = {"twist": RelativeHeadingCommandCfg(
        heading_law=opt.heading_law, heading_wz=opt.heading_wz, narrow_cols=narrow_cols,
        omni_mix=shares, omni_cols=omni_columns(subs) if shares else (),
        fwd_vx=fwd, omni_speed_max=tuple(opt.omni_speed_max), omni_wz=tuple(opt.omni_wz),
        spin_wz_max=tuple(opt.spin_wz_max), turn_wz_max=tuple(opt.turn_wz_max), ramp_iters=int(opt.cmd_ramp_iters),
        ramp_start_step=int(opt.cmd_ramp_start_step), fast_share=float(opt.cmd_fast_share),
        fast_speed=tuple(opt.fast_speed), fast_wz=tuple(opt.fast_wz), yaw_zero_share=float(opt.yaw_zero_share),
        entity_name="robot", resampling_time_range=tuple(opt.resample_s), heading_command=True,
        heading_control_stiffness=opt.heading_kp, rel_standing_envs=opt.rel_standing, rel_heading_envs=opt.rel_heading,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, opt.vx_max), lin_vel_y=(0.0, 0.0),
                                                ang_vel_z=(-opt.wz_max, opt.wz_max),
                                                heading=(-opt.heading_turn, opt.heading_turn)))}

    # events
    if narrow_cols:
        cfg.events["reset_base"] = EventTermCfg(func=reset_base_narrow, mode="reset", params={
            "pose_range": {"yaw": (-math.pi, math.pi)}, "narrow_cols": narrow_cols,
            "narrow_pose_range": {"x": (-0.03, 0.03), "y": (-0.01, 0.01), "yaw": (-math.radians(2), math.radians(2))}})
    else:
        cfg.events["reset_base"] = EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                                params={"pose_range": {"yaw": (-math.pi, math.pi)},
                                                        "velocity_range": {}})
    tiers = push_tiers(opt)
    if opt.randomise:
        if opt.foot_softness is not None:
            cfg.events["foot_softness"] = EventTermCfg(
                func=foot_softness, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                        "ranges": tuple(opt.foot_softness)})
        if tiers:
            cfg.events["push"] = EventTermCfg(func=PushKick, mode="step", params={
                "tiers": tiers, "asset_cfg": SceneEntityCfg("robot"), "start_steps": push_start_steps(opt),
                "grace_s": opt.push_fall_grace if opt.push_fall_grace > 0 else 2.0})
        elif opt.push_mps > 0:
            p = opt.push_mps
            cfg.events["push"] = EventTermCfg(func=envs_mdp.push_by_setting_velocity, mode="interval",
                                              interval_range_s=(3.0, 6.0),
                                              params={"velocity_range": {"x": (-p, p), "y": (-p, p)}})
    elif tiers:
        raise ValueError("kicks need randomise")

    # rewards (the walker's, in their order)
    r = {
        "track_lin_vel": RewardTermCfg(func=track_lin_vel_xy_exp, weight=2.0, params={"std": 0.10}),
        "track_ang_vel": RewardTermCfg(func=track_ang_vel_z_exp, weight=1.0, params={"std": 0.25}),
        "lin_vel_z": RewardTermCfg(func=lin_vel_z_l2, weight=-1.0),
        "ang_vel_xy": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.10),
        "torques": RewardTermCfg(func=mdp.torques_l2, weight=-1.0e-4),
        "joint_acc": RewardTermCfg(func=joint_acc_l2, weight=-1.0e-7),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
        "stance_slip": RewardTermCfg(func=stance_foot_slip_l2, weight=-2.0, params={"threshold": 1.0}),
    }
    if opt.alive:
        r["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=opt.alive)
    if opt.terminated:
        r["terminated"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=opt.terminated)
    r["upright"] = RewardTermCfg(func=upright_terrain_l2, weight=-1.0)
    if opt.clearance == "speed":
        r["foot_clearance"] = RewardTermCfg(func=foot_clearance_now, weight=-5.0, params={"target_height": 0.025 + fc})
    elif opt.clearance == "apex":
        r["swing_apex"] = RewardTermCfg(func=SwingApexClearance, weight=opt.clearance_weight,
                                        params={"target": opt.clearance_target, "foot_clear": fc})
    else:
        raise ValueError(opt.clearance)
    if opt.progress:
        r["progress"] = RewardTermCfg(func=progress_linear, weight=opt.progress)
    if opt.overspeed:
        if not shares:
            raise ValueError("overspeed acts on the omni mix's walks: needs omni_mix")
        r["overspeed"] = RewardTermCfg(func=overspeed_omni, weight=opt.overspeed)
    if opt.stumble:
        r["stumble"] = RewardTermCfg(func=feet_stumble, weight=opt.stumble)
    r["air_time"] = RewardTermCfg(func=vel_mdp.feet_air_time, weight=0.25,
                                  params={"sensor_name": "feet_contact", "threshold_min": 0.05,
                                          "threshold_max": 0.30, "command_name": "twist",
                                          "command_threshold": 0.02})
    r["leg_contact"] = RewardTermCfg(func=leg_contacts, weight=opt.leg_contact)
    if opt.abduction:
        r["abduction"] = RewardTermCfg(func=abduction_deviation_l2, weight=opt.abduction)
    r["stand_still"] = RewardTermCfg(func=stand_still_l2, weight=-1.0, params={})
    if opt.yaw_hold_weight:
        r["yaw_hold"] = RewardTermCfg(func=yaw_hold, weight=opt.yaw_hold_weight, params={"std": opt.yaw_hold_std})
    if opt.spin_progress_weight:
        if not shares:
            raise ValueError("spin_progress acts on the omni mix's spins and walk-and-turns: needs omni_mix")
        r["spin_progress"] = RewardTermCfg(func=spin_progress, weight=opt.spin_progress_weight)
    if opt.yaw_sign_stats:
        r["yaw_sign"] = RewardTermCfg(func=YawSignStats, weight=1.0e-9, params={})
    r["path"] = RewardTermCfg(func=PathTracker, weight=1.0e-9, params={"omni": True} if shares else {})
    cfg.rewards = r

    # terminations and curriculum
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(func=out_of_bounds, time_out=True,
                                                           params={"half_x": half_x, "half_y": half_y})
    cfg.terminations["tilt"].params["limit_rad"] = opt.tilt_limit
    if opt.band_deg is not None:
        band = tuple(float(b) for b in opt.band_deg)
        assert len(band) == 12 and all(b > 0 for b in band), band
        cfg.actions["servo"].band_deg = band
    servo_limits(cfg.actions["servo"], opt, "walk")
    if opt.stair_exit:
        cfg.terminations["stair_exit"] = TerminationTermCfg(func=stair_exit, time_out=True,
                                                            params={"subs": tuple(subs.values()), "tile": opt.tile})
        if narrow_cols:
            cfg.terminations["side_exit"] = TerminationTermCfg(func=side_exit, params={"subs": tuple(subs.values()),
                                                                                         "foot_clear": fc})
    if rows > 1:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=terrain_levels_path, params={
            "promote_m": opt.tile / 2, **({"omni_rel": opt.omni_promote_rel} if opt.omni_promote_rel else {}),
            **({"grace_s": opt.push_fall_grace} if opt.push_fall_grace > 0 else {})})
        if len(subs) > 1:
            cfg.curriculum["family_levels"] = CurriculumTermCfg(func=family_levels, params={"names": tuple(subs)})
    if "push" in cfg.events and cfg.events["push"].func is PushKick:
        cfg.curriculum["push_stats"] = CurriculumTermCfg(func=push_stats, params={})
    if opt.push_fall_grace and not tiers:
        raise ValueError("push_fall_grace needs a PushKick tier")
    if shares:
        cfg.curriculum["command_range"] = CurriculumTermCfg(func=command_range, params={})
    if opt.yaw_sign_stats:
        cfg.curriculum["yaw_sign"] = CurriculumTermCfg(func=yaw_sign_log, params={})
    cfg.sim.njmax = 600
    cfg.sim.nconmax = 64
    cfg.episode_length_s = opt.episode_s
    late_events_last(cfg)               # the attachment last: it builds on the plant's draws
    return cfg
