"""Score a get-up policy in simulation: from fixed fallen starts, does it stand up, how fast, and at what servo current.

    python -m mp4_rl.skills.getup.eval.getup_eval --policy mp4_rl/skills/getup/policies/sku2 --seed 12 --robots 256 --seconds 8 \
        --starts left,right,nose,back,midfall,crouch,random --motion --json runs/eval/getup_s12.json
    MUJOCO_GL=egl python -m mp4_rl.skills.getup.eval.getup_eval --policy mp4_rl/skills/getup/policies/sku2 --seed 12 --json OUT --video left.mp4 \
        --video-start left

The get-up task (``getup.task``) with its plant draws (friction, center of mass, mass, delay) seeded before the env
is built, so every start and every policy meets the same simulated robots; the observation noise at its end amplitudes;
the band and action scale from the policy's ``config.json``. Starts (one batch of ``--robots`` per start, yaw random,
joints the stand + U(+-0.5 band), base 0.06-0.10 m above the floor, drawn again where the spawn overlaps): left = roll +90
deg, right = roll -90, back = roll 180, nose = pitch +75 (nose down), random = the training's full start mix; crouch,
partial, side, midfall = those training slices, drawn as in training. Each trial runs ``--seconds`` (6).

Per trial: success (tilt < 12 deg, |gyro| < 0.8 rad/s, joint rms < 10 deg from the stand, held 0.5 s) and the time it
completed; ``success_with_height`` (the training reward's: also the base above 0.05 m); the time any servo is above 0.8 A,
the peak, and simulated guard trips (a servo above 1 A for 0.5 s) with the currents at x1 and x1.7 (the robot's currents
read up to 1.7x BAM's on one joint); the deepest floor and self penetration over the trial, after the spawn's drop has
landed (``SETTLE_S``), and sustained: the deepest penetration one geom pair holds on every step of ``SUSTAIN_S`` (0.2 s)
after ``SETTLE_S`` (impacts sink a body for a few steps). The JSON holds per-trial arrays and a per-start summary.
``--motion``: the motion quality and impact of every robot (``motion_meter.MotionMeter``), in each start's summary and per
trial. ``--reach``: each leg's commanded and measured hip towards the body's middle,
over all the starts after ``SETTLE_S``, against the robot's proven reach (``getup/eval/reach.py``): the JSON's ``reach`` and what's
beyond it printed.

On terrain (``run_terrain_start``; 15 s by default, the runner's retry on):

    python -m mp4_rl.skills.getup.eval.getup_eval --policy mp4_rl/skills/getup/policies/sku2 --seed 12 --robots 2048 --terrain low_poly,boxes \
        --levels 3,6 --json runs/eval/rough.json
    python -m mp4_rl.skills.getup.eval.getup_eval --policy mp4_rl/skills/getup/policies/sku2 --seed 12 --robots 1024 --terrain stairs4 --json OUT

``--terrain`` names rough families (``getup.task.ROUGH_FAMILIES``, "flat") or printed stairs (``stairs4``, ``stairs8``). Robots
drop at random positions (``terrain_spawn``), shared evenly among family, level and tile. Up: ``goal_now``, held 0.5 s (rough
ground: tilt < 40 deg on stairs and < 25 elsewhere, joint rms < 25 deg, gyro < 1.5 rad/s, three feet in contact). A robot
that resets mid-trial (a physics blow-up) is a failure whatever it did before or after, and is counted in ``resets``. The
retry restarts the phase clock when it has been at its end for ``--getup-phase-restart`` s and the tilt is above 45 deg. The
JSON holds per-trial arrays and the summary per start, per family and per family and level.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mp4_sim.actuators.choice import add_args as add_servo_args
from mp4_sim.actuators.choice import options_from_args
from mp4_sim.envs import terrain_spawn
from mp4_sim.envs.attachment import Attachment
from mp4_sim.envs.terrain import LEVELS, PRINTED_STAIRS, family_of
from mp4_sim.robot.model import legs_meta, legs_of

from mp4_rl.common import built_env
from mp4_rl.common.bounds import add_legs_args, eval_legs, eval_plant, legs_flags
from mp4_rl.common.env import noise_profile
from mp4_rl.common.mdp import servo_action
from mp4_rl.deploy.runner.runner import PHASE_RESTART_TILT_DEG
from mp4_rl.skills.getup.eval import reach as reach_gate
from mp4_rl.skills.getup.eval.motion_meter import MotionMeter
from mp4_rl.skills.getup.task import (
    EXACT,
    ROUGH_FAMILIES,
    SLICES,
    SUCCESS_HOLD_S,
    GetupOptions,
    GetupStart,
    _robot_geom_mask,
    base_z,
    contact_penetration,
    goal_now,
    make_getup_env_cfg,
    pair_penetration,
    policy_filter_alpha_of,
    success_now,
    tilt_deg,
    track_z,
)
from mp4_rl.skills.walker.eval.terrain import TorchPolicy, bam_currents

DT = 0.02
STARTS = ("left", "right", "back", "nose", "random")
SLICE_STARTS = ("crouch", "partial", "side", "midfall")   # training slices (getup_task.SLICES), drawn as in training
HIGH_A = 0.8
GUARD_A, GUARD_S = 1.0, 0.5
ROBOT_RATIO = 1.7
SETTLE_S = 0.5          # the spawn's drop (0.06-0.10 m) lands within this
SUSTAIN_S = 0.2         # one geom pair's depth held this long, after SETTLE_S
TERRAIN_STARTS = ("left", "right", "nose", "back", "midfall")
TERRAIN_SECONDS = 15.0
TILES_PER_CELL = 4      # tiles of each family and level, so a result is not one tile's
SUNK_M = 0.012          # a base this close to the ground under it has gone through it


def start_term(env) -> GetupStart:
    for cfg in env.event_manager._mode_class_term_cfgs["reset"]:
        if isinstance(cfg.func, GetupStart):
            return cfg.func
    raise KeyError("no GetupStart reset term")


def set_start(env, name: str) -> None:
    """Every env starts as ``name`` at the next reset: an EXACT key (one orientation, yaw random), a training slice
    (``getup.task.SLICES`` but "random": drawn as in training), or "random" (the full mix)."""
    term = start_term(env)
    n, dev = env.num_envs, env.device
    term.fixed_slice = None
    if name in SLICE_STARTS:
        term._exact = None
        term._fixed_random = None
        term.fixed_slice = torch.full((n,), SLICES.index(name), dtype=torch.long, device=dev)
        term.start_name = [name] * n
        return
    ex = torch.full((n, 2), float("nan"), device=dev)
    if name != "random":
        ex[:] = torch.tensor(EXACT[name], device=dev)
    term._exact = ex
    term._fixed_random = torch.full((n,), name == "random", dtype=torch.bool, device=dev)
    term.start_name = [name] * n


def build(policy_dir: Path, robots: int, seed: int, render: bool = False, contact_solref="run", limits=None,
          filter_alpha=None, device: str = "cuda:0", terrain: dict | None = None, attachment=None,
          actuator: dict | None = None, legs: dict | None = None, masses_json: str | None = None):
    """``masses_json``: the plant's masses file (None: the run's option). ``contact_solref``: "run" = the policy's run's. ``terrain``: more ``GetupOptions`` (``terrain_kwargs``).
    ``limits``: the measured joint limits (None: the run's option, else the policy's meta). ``filter_alpha``: the
    decoder's target filter (None: the policy's meta ``action_filter_alpha``, else the run's option, else 0).
    ``device``: the env's (``getup_stream`` builds one robot on the CPU). The widening of the joint limits is the
    policy's (its meta's bounds, ``limits.eval_plant``). ``attachment``: on every robot (none unless given, whatever the
    policy trained with). ``actuator``: the servo flags (``servo_model.options_from_args``); the model defaults to the
    policy's meta's ``actuator``, else the run's option, else md300; the supply and winding randomisation are off unless
    given. ``legs``: the flags (``limits.legs_flags``) over the policy's meta's, else the run's options'
    (``limits.eval_legs``)."""
    cfg_json = json.loads((policy_dir / "config.json").read_text())
    o = cfg_json["task"]["options"]
    meta = json.loads(str(np.load(policy_dir / "policy.npz")["meta"]))
    if limits is None:
        limits = bool(o.get("limits", False) or meta.get("limits", False))
    if filter_alpha is None:
        filter_alpha = policy_filter_alpha_of(o, meta)
    src = meta if meta.get("legs") else {"legs": {k: o.get(k) for k in ("legs", "foot", "foot_radius_m", "robot")},
                                         "display_collision": bool(o.get("display_collision", False))}
    noise_profile(cfg_json.get("noise", "measured"))       # refuses a noise profile other than the measured one
    opt = GetupOptions(num_envs=robots, noisy=True, noise_ramp=False, randomise=True, band_deg=tuple(o["band_deg"]),
                       action_scale=tuple(o["action_scale"]), masses_json=masses_json or o.get("masses_json"),
                       getup_collisions=o.get("getup_collisions", True), thigh_base=o.get("thigh_base", True),
                       thigh_short_m=o.get("thigh_short_m", 0.008), fixed_starts=("random",), visuals=render,
                       contact_solref=contact_solref if contact_solref != "run" else o.get("contact_solref"),
                       episode_s=1.0e3, phase_s=o.get("phase_s", 0.0) or 0.0, limits=bool(limits),
                       widen_limits_deg=eval_plant(meta)["widen_limits_deg"], filter_alpha=float(filter_alpha),
                       attachment=attachment, **servo_options(meta, o, actuator), **eval_legs(src, **(legs or {})),
                       **(terrain or {}))
    cfg = make_getup_env_cfg(opt)
    cfg.seed = seed
    if render:
        cfg.viewer.width, cfg.viewer.height = 960, 540
        cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 0.5, -20.0, 135.0
        cfg.viewer.env_idx, cfg.viewer.max_extra_envs = 0, 0
    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None), opt


def servo_options(meta: dict, run_options: dict, flags: dict | None = None) -> dict:
    """The servo model and its randomisation for an evaluation (``GetupOptions`` fields): the flags, else the policy's
    meta's ``actuator``, else the run's option, else md300; randomisation only from the flags."""
    flags = dict(flags or {})
    if flags.get("actuator") is None:
        flags["actuator"] = meta.get("actuator") or run_options.get("actuator") or None
    return {k: v for k, v in eval_plant(meta, actuator={k: v for k, v in flags.items() if v is not None}).items()
            if k in ("actuator", "supply_v_range", "supply_drop_ohm_range", "r_scale_range")}


def servo_record(opt) -> dict:
    """What a result file names about the servo model."""
    return {"actuator": opt.actuator, "supply_v_range": opt.supply_v_range,
            "supply_drop_ohm_range": opt.supply_drop_ohm_range, "r_scale_range": opt.r_scale_range}


def window_stats(cmax_steps, after_steps, trips_win, t_succ, seconds) -> dict:
    """Current before and after the first success (a trial that never succeeds is all 'before'): the p99 of the
    per-step largest servo current over the window's samples, and the share of trials with a simulated guard trip
    inside the window (x1, x1.7)."""
    if not cmax_steps:
        return {}
    c, a = torch.stack(cmax_steps), torch.stack(after_steps)              # (steps, n)
    out = {"time_before_success_mean_s": float(torch.where(t_succ >= 0, t_succ, torch.full_like(t_succ, seconds))
                                               .mean())}
    for w, mask in (("before", ~a), ("after", a)):
        v = c[mask]
        out[f"current_p99_{w}_A"] = float(v.quantile(0.99)) if v.numel() else None
        out[f"guard_trips_x1_{w}"] = float(trips_win[(1.0, w)].float().mean())
        out[f"guard_trips_x1p7_{w}"] = float(trips_win[(ROBOT_RATIO, w)].float().mean())
    return out


def goal_criteria(env):
    """(the success criterion, with the height, without the gyro) per env: tilt, gyro, joint rms; + the base above 0.05
    m; tilt, rms and height without the gyro."""
    ok = success_now(env)
    up = base_z(env) > track_z(env)
    return ok, ok & up, success_now(env, None) & up


def run_start(env, pol, name: str, seconds: float, seed: int, frames: list | None = None,
              noise: float = 0.0, meter: MotionMeter | None = None, reach: reach_gate.ReachMeter | None = None) -> dict:
    set_start(env, name)
    torch.manual_seed(seed)
    obs, _ = env.reset()
    if meter is not None:
        meter.begin(env)
    n, dev = env.num_envs, env.device
    steps = int(round(seconds / DT))
    hold = int(round(SUCCESS_HOLD_S / DT))
    run = torch.zeros(n, dtype=torch.long, device=dev)
    t_succ = torch.full((n,), -1.0, device=dev)
    run_h = torch.zeros(n, dtype=torch.long, device=dev)      # the training's success: + the base above 0.05 m
    t_succ_h = torch.full((n,), -1.0, device=dev)
    run_ng = torch.zeros(n, dtype=torch.long, device=dev)     # tilt, rms and height, no gyro condition
    t_succ_ng = torch.full((n,), -1.0, device=dev)
    tilt_after = torch.zeros(n, device=dev)
    high = torch.zeros(n, device=dev)
    high_after = torch.zeros(n, device=dev)
    peak = torch.zeros(n, device=dev)
    guard = {k: torch.zeros(n, 12, device=dev) for k in (1.0, ROBOT_RATIO)}       # time above 1 A, per servo
    trips = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in guard}
    trips_win = {(k, w): torch.zeros(n, dtype=torch.bool, device=dev) for k in guard for w in ("before", "after")}
    cmax_steps, after_steps = [], []                   # per step: the largest servo current, success completed
    pen_floor = torch.zeros(n, device=dev)
    pen_self = torch.zeros(n, device=dev)
    pen_late = torch.zeros(n, device=dev)          # floor or self, after the spawn's drop (SETTLE_S)
    settle = int(round(SETTLE_S / DT))
    hold_pen = int(round(SUSTAIN_S / DT))
    window = []                                    # the last hold_pen steps' per-pair penetration
    robot_g = _robot_geom_mask(env)
    selfpair = robot_g[:, None] & robot_g[None, :]
    sus = torch.zeros(n, device=dev)               # deepest penetration of one geom pair held SUSTAIN_S, after settle
    sus_self = torch.zeros(n, device=dev)
    term = start_term(env)
    tilt0 = tilt_deg(env).clone()
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    for k in range(steps):
        act = pol(obs["actor"])
        if noise > 0.0:          # the training's exploration: Gaussian action noise, std in action units
            act = act + noise * torch.randn(act.shape, device=act.device, generator=gen)
        obs, _, _, _, _ = env.step(act)
        if meter is not None:
            meter.update(env)
        if reach is not None and k >= settle:             # the spawn's own pose isn't the policy's
            reach.add_servo(servo_action(env), torch.ones(n, dtype=torch.bool, device=dev))
        if frames is not None:
            frames.append(env.render())
        ok, ok_h, ok_ng = goal_criteria(env)
        run = torch.where(ok, run + 1, torch.zeros_like(run))
        hit = (run >= hold) & (t_succ < 0)
        t_succ = torch.where(hit, torch.full_like(t_succ, (k + 1) * DT), t_succ)
        run_h = torch.where(ok_h, run_h + 1, torch.zeros_like(run_h))
        t_succ_h = torch.where((run_h >= hold) & (t_succ_h < 0), torch.full_like(t_succ_h, (k + 1) * DT), t_succ_h)
        run_ng = torch.where(ok_ng, run_ng + 1, torch.zeros_like(run_ng))
        t_succ_ng = torch.where((run_ng >= hold) & (t_succ_ng < 0), torch.full_like(t_succ_ng, (k + 1) * DT),
                                t_succ_ng)
        tilt_after = torch.where(t_succ >= 0, torch.maximum(tilt_after, tilt_deg(env)), tilt_after)
        c = bam_currents(env)
        if c is not None:
            over = (c > HIGH_A).any(-1).float() * DT
            high += over
            high_after += over * (t_succ >= 0).float()            # after the success criterion completed
            peak = torch.maximum(peak, c.max(-1).values)
            cmax_steps.append(c.max(-1).values)
            after_steps.append(t_succ >= 0)
            for r in guard:
                guard[r] = torch.where(c * r > GUARD_A, guard[r] + DT, torch.zeros_like(guard[r]))
                trip = (guard[r] >= GUARD_S - 1e-6).any(-1)
                trips[r] |= trip
                trips_win[(r, "before")] |= trip & (t_succ < 0)
                trips_win[(r, "after")] |= trip & (t_succ >= 0)
        f, s = contact_penetration(env)
        pen_floor = torch.maximum(pen_floor, f)
        pen_self = torch.maximum(pen_self, s)
        if k >= settle:
            pen_late = torch.maximum(pen_late, torch.maximum(f, s))
            window.append(pair_penetration(env))
            if len(window) > hold_pen:
                window.pop(0)
            if len(window) == hold_pen:
                held = torch.stack(window).amin(0)                     # (n, G, G): exceeded on every step
                sus = torch.maximum(sus, held.flatten(1).amax(-1))
                sus_self = torch.maximum(sus_self, (held * selfpair).flatten(1).amax(-1))
    succ = (t_succ >= 0).cpu().numpy()
    t = t_succ.cpu().numpy()
    trial = {"success": succ.tolist(), "t_success": t.tolist(),
             "success_with_height": (t_succ_h >= 0).cpu().numpy().tolist(),
             "success_no_gyro": (t_succ_ng >= 0).cpu().numpy().tolist(), "tilt0_deg": tilt0.cpu().numpy().round(2).tolist(),
             "slice": term.slice_id.cpu().numpy().tolist(), "spawn_tries": term.tries.cpu().numpy().tolist(),
             "max_tilt_after_success_deg": tilt_after.cpu().numpy().round(2).tolist(),
             "time_above_0p8A_s": high.cpu().numpy().round(3).tolist(),
             "time_above_0p8A_after_success_s": high_after.cpu().numpy().round(3).tolist(), "peak_A": peak.cpu().numpy().round(4).tolist(),
             "guard_trip_x1": trips[1.0].cpu().numpy().tolist(),
             "guard_trip_x1p7": trips[ROBOT_RATIO].cpu().numpy().tolist(),
             "pen_floor_mm": (pen_floor * 1000).cpu().numpy().round(3).tolist(),
             "pen_self_mm": (pen_self * 1000).cpu().numpy().round(3).tolist(),
             "pen_after_settle_mm": (pen_late * 1000).cpu().numpy().round(3).tolist(),
             "pen_sustained_mm": (sus * 1000).cpu().numpy().round(3).tolist(),
             "pen_sustained_self_mm": (sus_self * 1000).cpu().numpy().round(3).tolist()}
    ta = tilt_after.cpu().numpy()[succ]
    summary = {"start": name, "n": int(n), "success": float(succ.mean()),
               "success_with_height": float((t_succ_h >= 0).float().mean()),
               "success_no_gyro": float((t_succ_ng >= 0).float().mean()),
               "t_success_p50": float(np.median(t[succ])) if succ.any() else None,
               "t_success_p90": float(np.percentile(t[succ], 90)) if succ.any() else None,
               "max_tilt_after_success_p90_deg": float(np.percentile(ta, 90)) if succ.any() else None,
               "time_above_0p8A_mean_s": float(high.mean()), "time_above_0p8A_p90_s": float(high.quantile(0.9)),
               "time_above_0p8A_after_success_mean_s": float(high_after.mean()),
               "time_above_0p8A_before_success_mean_s": float((high - high_after).mean()),
               **window_stats(cmax_steps, after_steps, trips_win, t_succ, seconds),
               "time_after_success_mean_s": float(((seconds - t_succ) * (t_succ >= 0).float()).mean()),
               "peak_A_p50": float(peak.median()), "peak_A_max": float(peak.max()),
               "guard_trips_x1": float(trips[1.0].float().mean()),
               "guard_trips_x1p7": float(trips[ROBOT_RATIO].float().mean()),
               "pen_floor_mm_max": float(pen_floor.max() * 1000), "pen_self_mm_max": float(pen_self.max() * 1000),
               "pen_mm_p99": float(torch.maximum(pen_floor, pen_self).quantile(0.99) * 1000),
               "pen_after_settle_mm_max": float(pen_late.max() * 1000),
               "pen_after_settle_mm_p99": float(pen_late.quantile(0.99) * 1000),
               "pen_sustained_mm_max": float(sus.max() * 1000), "pen_sustained_mm_p99": float(sus.quantile(0.99) * 1000),
               "pen_sustained_self_mm_max": float(sus_self.max() * 1000),
               "spawn_unresolved": int(term.unresolved)}
    if meter is not None:
        m = meter.finish(t)
        summary["motion"] = m["summary"]
        trial["motion"] = m["trials"]
    return {"summary": summary, "trials": trial}


def terrain_kwargs(families: list[str], levels: list[int], relief_pitch: float | None = None) -> dict:
    """``GetupOptions`` for ``families`` up to the highest of ``levels``; every tile unseen in training."""
    rows = max(levels) + 1
    columns = [f for f in families for _ in range(1 if f in PRINTED_STAIRS or f == "flat" else TILES_PER_CELL)]
    return {"terrain": "families", "families": tuple(columns), "rows": rows, "difficulty_max": rows / LEVELS,
            "terrain_seed": 1, "relief_pitch": relief_pitch}


def terrain_cells(columns: list[str], levels: list[int]) -> list[tuple[int, int]]:
    """(row, column) of every family, level and tile, so ``i % len`` shares robots evenly among families and levels."""
    fams = list(dict.fromkeys(family_of(c) for c in columns))
    per = {f: [i for i, c in enumerate(columns) if family_of(c) == f] for f in fams}
    n = max(len(v) for v in per.values())
    return [(lv, per[f][k % len(per[f])]) for f in fams for lv in levels for k in range(n)]


class UpTracker:
    """Per robot, the time it first held a rule for ``hold`` steps. A robot that resets is out for good: its later steps
    are another robot's attempt, and an earlier success is forgotten."""

    def __init__(self, n: int, hold: int, device):
        self.hold = hold
        self.run = torch.zeros(n, dtype=torch.long, device=device)
        self.t_up = torch.full((n,), -1.0, device=device)
        self.out = torch.zeros(n, dtype=torch.bool, device=device)

    def update(self, ok: torch.Tensor, reset: torch.Tensor, t: float) -> None:
        self.out |= reset
        self.run = torch.where(ok & ~self.out, self.run + 1, torch.zeros_like(self.run))
        self.t_up = torch.where((self.run >= self.hold) & (self.t_up < 0), t, self.t_up)
        self.t_up = torch.where(self.out, -1.0, self.t_up)


def run_terrain_start(env, pol, name: str, seconds: float, seed: int, restart_after_s: float, step_hook=None) -> dict:
    """One batch from start ``name``. ``restart_after_s``: the retry's phase-clock restart (0: none). ``step_hook(k)``
    runs before step k. Returns per-trial numpy arrays."""
    spawn = env.terrain_spawn
    set_start(env, name)
    torch.manual_seed(seed)
    obs, _ = env.reset()
    n, dev = env.num_envs, env.device
    robot = env.scene["robot"]
    origin = env.scene.env_origins.clone()
    hold = int(round(SUCCESS_HOLD_S / DT))
    up, strict = UpTracker(n, hold, dev), UpTracker(n, hold, dev)
    phase_k = torch.zeros(n, dtype=torch.long, device=dev)
    restarts = torch.zeros(n, dtype=torch.long, device=dev)
    peak = torch.zeros(n, device=dev)
    for k in range(int(round(seconds / DT))):
        if step_hook is not None:
            step_hook(k)
        obs, _, term, trunc, _ = env.step(pol(obs["actor"]))
        reset = term | trunc
        t = (k + 1) * DT
        up.update(goal_now(env), reset, t)
        strict.update(success_now(env), reset, t)
        c = bam_currents(env)
        if c is not None:
            peak = torch.maximum(peak, c.max(-1).values * (up.t_up < 0).float())
        if restart_after_s > 0.0:
            due = ((k - phase_k).float() * DT >= restart_after_s - 1e-9) & (up.t_up < 0) & ~up.out
            due &= tilt_deg(env) > PHASE_RESTART_TILT_DEG
            if bool(due.any()):
                env.episode_length_buf[due] = 0
                phase_k[due] = k
                restarts[due] += 1
    p = robot.data.root_link_pos_w
    ground = spawn.ground
    return {"t_up": up.t_up.cpu().numpy(), "t_strict": strict.t_up.cpu().numpy(), "reset": up.out.cpu().numpy(),
            "restarts": restarts.cpu().numpy(), "final_tilt_deg": tilt_deg(env).cpu().numpy(),
            "row": spawn.row.cpu().numpy(), "col": spawn.col.cpu().numpy(), "peak_A": peak.cpu().numpy(),
            "drop_m": (origin[:, 2] - ground.under(origin[:, :2])).cpu().numpy(),
            "moved_m": (p[:, :2] - origin[:, :2]).norm(dim=-1).cpu().numpy(),
            "height_m": (p[:, 2] - ground.under(p[:, :2])).cpu().numpy()}


def terrain_summary(tr: dict, mask: np.ndarray, first_try_s: float) -> dict:
    """A summary of the trials ``mask``. ``never_up`` splits into the resets and, by the final tilt, the rest."""
    t, reset, tilt = tr["t_up"][mask], tr["reset"][mask], tr["final_tilt_deg"][mask]
    up = t >= 0
    down = ~up & ~reset
    return {"n": int(mask.sum()), "up": float(up.mean()), "strict": float((tr["t_strict"][mask] >= 0).mean()),
            "up_first_try": float(((t >= 0) & (t <= first_try_s)).mean()),
            "t_up_p50": float(np.median(t[up])) if up.any() else None,
            "never_up": int((~up).sum()), "resets": int(reset.sum()), "down_on_back": int((down & (tilt > 120)).sum()),
            "down_on_side": int((down & (tilt > PHASE_RESTART_TILT_DEG) & (tilt <= 120)).sum()),
            "down_low_tilt": int((down & (tilt <= PHASE_RESTART_TILT_DEG)).sum()),
            "restarts_mean": float(tr["restarts"][mask].mean()), "peak_A_p90": float(np.percentile(tr["peak_A"][mask], 90)),
            "drop_mm_p50": float(np.median(tr["drop_m"][mask]) * 1000),
            "drop_mm_p90": float(np.percentile(tr["drop_m"][mask], 90) * 1000),
            "moved_m_p90": float(np.percentile(tr["moved_m"][mask], 90)),
            "sunk": int((tr["height_m"][mask] < SUNK_M).sum())}


def terrain_main(a, pdir: Path, pol, contact_solref) -> int:
    families = [f for f in a.terrain.split(",") if f]
    known = ROUGH_FAMILIES + tuple(PRINTED_STAIRS) + ("flat",)
    assert all(f in known for f in families), (families, known)
    levels = [int(v) for v in a.levels.split(",")]
    if all(f in PRINTED_STAIRS for f in families):      # the same flight at every level
        levels = [0]
    seconds = a.seconds if a.seconds is not None else TERRAIN_SECONDS
    starts = [s for s in (a.starts or ",".join(TERRAIN_STARTS)).split(",") if s]
    assert all(s in STARTS + SLICE_STARTS for s in starts), starts
    env, opt = build(pdir, a.robots, a.seed, contact_solref=contact_solref, limits=a.limits, filter_alpha=a.filter_alpha,
                     terrain=terrain_kwargs(families, levels, a.relief_pitch), attachment=a.attachment,
                     actuator=options_from_args(a), legs=legs_flags(a), masses_json=a.masses_json)
    spawn = env.terrain_spawn
    cells = terrain_cells(spawn.columns, levels)
    k = torch.arange(a.robots) % len(cells)
    table = torch.tensor(cells)[k].to(env.device)
    spawn.fix_cells(table[:, 0], table[:, 1])
    restart = float(opt.phase_s) + a.getup_phase_restart if opt.phase_s else 0.0
    first_try = restart or seconds
    col_family = np.array(spawn.families)
    built = built_env.record(env, opt)
    print(built_env.line(built), flush=True)
    out = {"built_env": built, "masses_json": a.masses_json, "policy": str(pdir), "seed": a.seed, "robots": a.robots, "seconds": seconds, "levels": levels,
           "families": families, "phase_restart_after_s": restart, "limits": bool(opt.limits), **servo_record(opt),
           "legs_geometry": legs_meta(legs_of(opt)),
           "relief_pitch": opt.relief_pitch,
           "up_rule": {"tilt_stairs_deg": terrain_spawn.UP_TILT_STAIRS_DEG, "tilt_deg": terrain_spawn.UP_TILT_DEG,
                       "rms_deg": terrain_spawn.UP_RMS_DEG, "gyro": terrain_spawn.UP_GYRO, "feet": terrain_spawn.UP_FEET,
                       "hold_s": SUCCESS_HOLD_S}, "starts": {}, "by_family": {}}
    all_trials = []
    for s in starts:
        tr = run_terrain_start(env, pol, s, seconds, a.seed, restart)
        fam = col_family[tr["col"]]
        all_trials.append({**tr, "family": fam})
        res = {"summary": terrain_summary(tr, np.ones(a.robots, dtype=bool), first_try), "by_family": {},
               "trials": {k_: np.round(v, 4).tolist() if v.dtype.kind == "f" else v.tolist() for k_, v in tr.items()}}
        for f in families:
            res["by_family"][f] = terrain_summary(tr, fam == f, first_try)
            print(s, f, json.dumps(res["by_family"][f]), flush=True)
        out["starts"][s] = res
    joined = {k_: np.concatenate([t[k_] for t in all_trials]) for k_ in all_trials[0]}
    out["summary"] = terrain_summary(joined, np.ones(len(joined["t_up"]), dtype=bool), first_try)
    for f in families:
        m = joined["family"] == f
        out["by_family"][f] = {"all": terrain_summary(joined, m, first_try),
                               "levels": {str(lv): terrain_summary(joined, m & (joined["row"] == lv), first_try)
                                          for lv in levels}}
        print("family", f, json.dumps(out["by_family"][f]["all"]), flush=True)
    env.close()
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="an exported policy directory (policy.npz + config.json)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--robots", type=int, default=256, help="robots per start (<= 1024)")
    ap.add_argument("--starts", help=f"default {','.join(STARTS)}; on terrain {','.join(TERRAIN_STARTS)}")
    ap.add_argument("--seconds", "--trial-s", dest="seconds", type=float,
                    help=f"s per trial (default 6; on terrain {TERRAIN_SECONDS:g})")
    ap.add_argument("--terrain", metavar="FAMILY[,FAMILY]", help="drop on these rough families or printed stairs "
                    "(stairs4, stairs8) at random positions, with the runner's retry")
    ap.add_argument("--levels", default="3,6", help="--terrain: difficulty levels (0-9)")
    ap.add_argument("--getup-phase-restart", type=float, default=1.0, metavar="S",
                    help="--terrain: the retry, S s after the phase clock reached its end (the runner's option)")
    ap.add_argument("--relief-pitch", type=float, metavar="M", help="--terrain: the reliefs' heightfield pitch")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="add N(0, this) to the policy's actions every step (the training's exploration noise)")
    ap.add_argument("--video", help="mp4: one robot (env 0) of --video-start, rendered in its own env")
    ap.add_argument("--video-start", default="left")
    ap.add_argument("--contact-solref", nargs="+", default=["run"],
                    help="the non-foot robot contacts' solref: 'run' (the policy's run's), 'none' (MuJoCo's default) or "
                         "TIMECONST DAMPRATIO (a comparison; the policy was trained with its run's)")
    ap.add_argument("--limits", action="store_true", default=None,
                    help="the measured joint limits (default: the run's option or the policy's meta)")
    ap.add_argument("--no-limits", dest="limits", action="store_false")
    ap.add_argument("--filter-alpha", type=float, metavar="X",
                    help="the decoder's target filter alpha (default: the policy's meta action_filter_alpha, else the "
                         "run's option filter_alpha, else 0)")
    ap.add_argument("--masses-json", help="the plant's masses file (a file name in the robot's measured data in mp4_sim or a path; "
                                          "default the run's)")
    ap.add_argument("--attachment", type=Attachment.parse, metavar="HEIGHT_MM,RADIUS_MM,X_M,MASS_G",
                    help="a fixed attachment on every robot: a cylinder on the hood and its mass")
    ap.add_argument("--motion", action="store_true",
                    help="measure motion quality and impact (motion_meter) into each start's summary and trials")
    add_legs_args(ap, "the policy's meta, else its run's options, else sku_2")
    reach_gate.add_args(ap)
    add_servo_args(ap, evaluation=True)
    a = ap.parse_args()
    pdir = Path(a.policy)
    pol = TorchPolicy(pdir / "policy.npz", "cuda:0")
    cs = a.contact_solref
    cs = "run" if cs == ["run"] else None if cs == ["none"] else tuple(float(v) for v in cs)
    if a.terrain:
        assert not (a.video or a.motion or a.action_noise), "--terrain takes no --video, --motion or --action-noise"
        assert a.robots <= 4096, "keep a job at <= 4096 envs"
        return terrain_main(a, pdir, pol, cs)
    a.seconds = 6.0 if a.seconds is None else a.seconds
    starts = [s for s in (a.starts or ",".join(STARTS)).split(",") if s]
    assert all(s in STARTS + SLICE_STARTS for s in starts), starts
    assert a.robots <= 1024, "keep a job at <= 1024 envs"
    env, opt = build(pdir, a.robots, a.seed, contact_solref=cs, limits=a.limits, filter_alpha=a.filter_alpha,
                     attachment=a.attachment, actuator=options_from_args(a), legs=legs_flags(a), masses_json=a.masses_json)
    built = built_env.record(env, opt)
    print(built_env.line(built), flush=True)
    out = {"built_env": built, "masses_json": a.masses_json, "policy": str(pdir), "seed": a.seed, "robots": a.robots, "seconds": a.seconds,
           "band_deg": list(opt.band_deg), "action_scale": list(opt.action_scale),
           "contact_solref": opt.contact_solref, "limits": bool(opt.limits), **servo_record(opt),
           "legs_geometry": legs_meta(legs_of(opt)),
           "filter_alpha": opt.filter_alpha, "action_noise": a.action_noise, "sustain_s": SUSTAIN_S, "settle_s": SETTLE_S,
           "attachment": asdict(a.attachment) if a.attachment else None,
           "current_sampling": "per policy step (20 ms), |tau|/kt of the last physics step",
           "guard": {"amps": GUARD_A, "seconds": GUARD_S, "ratios": [1.0, ROBOT_RATIO]}, "starts": {}}
    meter = MotionMeter(env) if a.motion else None
    if meter is not None:
        out["motion"] = meter.config()
    reach = reach_gate.ReachMeter(reach_gate.load_proven(a.reach), robot=opt.robot) if a.reach else None
    for s in starts:
        r = run_start(env, pol, s, a.seconds, a.seed, noise=a.action_noise, meter=meter, reach=reach)
        out["starts"][s] = r
        print(json.dumps({k: v for k, v in r["summary"].items() if k != "motion"}), flush=True)
        if meter is not None:
            g = r["summary"]["motion"]["getup"]
            print("motion", s, json.dumps({k: g[k]["mean"] for k in ("rev5", "rev10", "pow3", "da2", "speed_rms")}),
                  flush=True)
    if reach is not None:
        out["reach"] = reach.report()
        reach_gate.print_report(out["reach"])
    if meter is not None:
        meter.detach()
    env.close()
    if a.video:
        import imageio.v2 as imageio
        venv, _ = build(pdir, 1, a.seed, render=True, contact_solref=cs, limits=opt.limits,
                        filter_alpha=opt.filter_alpha, attachment=a.attachment, actuator=options_from_args(a),
                        legs=legs_flags(a), masses_json=a.masses_json)
        frames = []
        r = run_start(venv, pol, a.video_start, a.seconds, a.seed, frames=frames)
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(a.video, frames, fps=int(1 / DT))
        out["video"] = {"path": a.video, "start": a.video_start, "summary": r["summary"], "trial": r["trials"]}
        print("video", a.video, json.dumps(r["summary"]), flush=True)
        venv.close()
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
