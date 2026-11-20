"""Get-up motions as files ``mp4link stream`` plays on a lifted robot: ``delta_dd`` (N x 12: each SET_ALL frame's
targets minus the contract's stand, in the firmware's deci-degrees, servo order) at ``period_s`` 0.02 s.

    .venv/bin/python tools/getup_stream.py --policy skills/getup/policies/sku2 --out-dir runs/streams

A learned get-up policy (``--policy DIR``, an export with ``config.json``) plays in one simulated robot in
``getup_eval``'s env (the policy's band, scale, limits, target filter alpha and plant), from a named start
(``getup_eval.STARTS``: back, left, right, nose, ...), ``--seed``; the applied targets the decoder made at every policy
step (the simulation's ``ServoAction``: band, 20 deg step limit, bounds, filter, the firmware's tick quantisation),
recorded from the start to ``--tail-s`` after the get-up's success (all ``--seconds`` if it never succeeds). The robot
side's own decoder (``policy.ActionDecoder``) is run on the same raw actions as a check
(``runner_decoder_max_dd_diff``). Each start is written at every ``--speeds`` factor (default 0.25 and 1.0): the
recorded targets resampled in time (linear in dd, 0.02 s frames, 1 / speed as many), so 0.25 moves a quarter as far per
frame. The plant flags (``--legs``, ``--foot``, ``--foot-radius``, ``--actuator`` and its supply options,
``--masses-json``) override the policy's (its meta's, else its run's); the summary records them and, per start, the
built env (``eval.built_env``: total mass, legs, servo, contract and limits hashes). A policy made for another
calibration than the contract's is refused, as the runner refuses it. Every frame is checked against the contract's
potentiometer dead zone (its stand dd plus the row at least ``--end-margin-deg``, default 20 as ``mp4link stream``'s,
from 0 and 3100 dd: refused, nothing written, otherwise).

(``--device cpu``, the default: one robot, about a minute per start; ``--device cuda:0`` with 3 GB free.) Files:
``<policy>_<start>_s<seed>_x<speed>.npz`` (``0p25``, ``1p00``) and ``<policy>_s<seed>.json`` (per file: frames,
seconds, success and its time, max deviation from the stand per joint, the margin to the track ends, the frames
``mp4link stream`` would change: beyond its band or more than its 20 deg per send).

On the robot (lifted, legs free): ``mp4link stream FILE --cap CAP --stop-c 55``, always in the firmware's get-up mode
and with the policy decoder's band and step (abduction 90 deg, hip and knee 180 deg, 20 deg per send). See
skills/getup/docs/ROBOT_SKU2.md. The JSON counts frames it would change (``frames_beyond_stream_band``, ``frames_over_stream_step``): 0,
or a frame or two by tick rounding (0.3 deg).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mp4_rl.common.policy import (
    DD0,
    TRACK_DD,
    ActionDecoder,
    Contract,
    calibration_refusal,
    policy_band_deg,
    policy_bounds,
    policy_stand_offset_deg,
)

PERIOD = 0.02
# mp4link stream: the band per servo (abduction 90 deg, hip and knee 180) and the step per send, as the decoder's
STREAM_BAND_MAX_DEG = np.array([90.0 if i % 3 == 0 else 180.0 for i in range(12)])
STREAM_STEP_DEG = 20.0
END_MARGIN_DEG = 20.0               # as mp4link stream (stream --end-margin-deg default)
POLICY_STARTS = ("back", "left", "right", "nose")


# ---- checks -------------------------------------------------------------------------------------------------------
def end_margin_deg(stand_dd, delta) -> np.ndarray:
    """Per servo the smallest distance (deg) of any frame's target (the stand plus the row) and the stand from the
    pot's track ends (0 and 3100 dd)."""
    tgt = np.vstack([np.asarray(stand_dd, float)[None], np.asarray(stand_dd, float)[None] + np.asarray(delta, float)])
    return np.minimum(tgt - TRACK_DD[0], TRACK_DD[1] - tgt).min(0) / 10.0


def dead_zone_refusal(stand_dd, delta, margin_deg: float = END_MARGIN_DEG) -> str | None:
    m = end_margin_deg(stand_dd, delta)
    if np.all(m >= margin_deg):
        return None
    i = int(np.argmin(m))
    return f"servo {i + 1}: a target {m[i]:.1f} deg from its track end (< {margin_deg:g})"


def stream_effects(delta) -> dict:
    """What ``mp4link stream`` would change in playing ``delta``: frames beyond its band (abduction 90 deg, hip
    and knee 180), frames moving more than its 20 deg per send. 0 for a learned get-up, up to a frame or two by tick rounding."""
    dev = np.abs(np.asarray(delta, float)) / 10.0
    step = np.abs(np.diff(np.asarray(delta, float), axis=0)) / 10.0
    return {"max_dev_deg": dev.max(0).round(2).tolist(),
            "frames_beyond_stream_band": int((dev > STREAM_BAND_MAX_DEG[None] + 0.05).any(-1).sum()),
            "max_step_deg": float(step.max()) if len(step) else 0.0,
            "frames_over_stream_step": int((step > STREAM_STEP_DEG + 0.05).any(-1).sum())}


def resample(dd, speed: float) -> np.ndarray:
    """Frames at 0.02 s of ``dd`` (N x 12, one per 0.02 s) played at ``speed`` (0.25: four times as long), linear in
    dd between the recorded frames, rounded to whole dd; the last frame kept."""
    dd = np.asarray(dd, dtype=float)
    if speed == 1.0:
        return dd.copy()
    t_src = np.arange(len(dd)) * PERIOD
    t = np.arange(0.0, t_src[-1] + 1e-9, PERIOD * speed)
    if t[-1] < t_src[-1] - 1e-9:
        t = np.append(t, t_src[-1])
    return np.rint(np.stack([np.interp(t, t_src, dd[:, i]) for i in range(12)], axis=1))


def speed_tag(speed: float) -> str:
    return f"x{speed:.2f}".replace(".", "p")


# ---- 1. a learned get-up policy ---------------------------------------------------------------------------------
def dd_of_quantised_rel(c: Contract, rel) -> np.ndarray:
    """The dd of an applied target already on the firmware's ticks (``Contract.quantise_rel``'s second value, the
    simulation's ``ServoAction`` target): the exact inverse of ``rel_from_dd`` (quantising it again could move it a
    tick: ``servo_deg_from_dd``)."""
    return np.rint(DD0 - 10.0 * (c.servo_deg_from_rel(rel) + c.offsets_deg))


def plant_kwargs(a) -> dict:
    """``getup_eval.build``'s plant arguments from the flags (``--legs``, ``--foot``, ``--foot-radius``, ``--actuator``
    and its supply options, ``--masses-json``); none given: the policy's meta's, else its run's."""
    from mp4_sim.actuators.choice import options_from_args

    from mp4_rl.common.bounds import legs_flags
    return {"actuator": options_from_args(a), "legs": legs_flags(a), "masses_json": getattr(a, "masses_json", None)}


def plant_flags_record(a) -> dict:
    """The plant flags as given (None: the policy's)."""
    return {"legs": a.legs, "foot": a.foot, "foot_radius_m": a.foot_radius, "actuator": a.actuator,
            "supply_v": a.supply_v, "supply_drop_ohm": a.supply_drop_ohm, "r_scale": a.r_scale,
            "masses_json": a.masses_json}


def record_policy(pdir: Path, start: str, seed: int, seconds: float, device: str, plant: dict | None = None) -> dict:
    """One simulated get-up: per step the applied target's dd (the simulation's decoder) and the runner decoder's dd
    on the same raw actions, the success time, the simulated currents. ``plant``: ``plant_kwargs``."""
    import torch

    from mp4_rl.common import built_env
    from mp4_rl.common.mdp import servo_action
    from mp4_rl.getup.eval import getup_eval as ge
    from mp4_rl.getup.task import SUCCESS_HOLD_S
    from mp4_rl.walker.eval.terrain import TorchPolicy, bam_currents

    c = Contract.load()
    pol = TorchPolicy(pdir / "policy.npz", device)
    meta = pol.meta
    env, opt = ge.build(pdir, 1, seed, device=device, **(plant or {}))
    built = built_env.record(env, opt)
    print(built_env.line(built), flush=True)
    term = servo_action(env)
    ge.set_start(env, start)
    torch.manual_seed(seed)
    obs, _ = env.reset()
    off = np.asarray(term.off.detach().cpu(), dtype=float).reshape(-1)[:12] if term._shifted else np.zeros(12)
    dec = ActionDecoder(c, np.asarray(pol.scale), band_deg=policy_band_deg(meta), bounds=policy_bounds(meta),
                        stand_offset_abd_deg=policy_stand_offset_deg(meta), filter_alpha=float(opt.filter_alpha))
    dec.prev_rel = np.asarray(term.prev_rel[0].detach().cpu(), dtype=np.float64)
    dec.prev_action = dec.prev_rel / dec.scale
    dd0 = dd_of_quantised_rel(c, dec.prev_rel + off)
    hold = int(round(SUCCESS_HOLD_S / PERIOD))
    run, t_succ = 0, None
    dd_sim, dd_run, cur = [], [], []
    for k in range(int(round(seconds / PERIOD))):
        with torch.no_grad():
            act = pol(obs["actor"])
        obs, _, _, _, _ = env.step(act)
        phys = np.asarray(term.prev_rel[0].detach().cpu(), dtype=np.float64) + off
        dd_sim.append(dd_of_quantised_rel(c, phys))
        dd_run.append(dec(np.asarray(act[0].detach().cpu(), dtype=np.float64))["dd"])
        a = bam_currents(env)
        cur.append(np.zeros(12) if a is None else np.asarray(a[0].detach().cpu(), dtype=float))
        ok = bool(ge.goal_criteria(env)[0][0])
        run = run + 1 if ok else 0
        if run >= hold and t_succ is None:
            t_succ = (k + 1) * PERIOD
    env.close()
    return {"dd0": np.asarray(dd0, float), "dd": np.asarray(dd_sim, float), "dd_runner": np.asarray(dd_run, float),
            "t_success": t_succ, "current_A": np.asarray(cur), "filter_alpha": float(opt.filter_alpha),
            "limits": bool(opt.limits), "band_deg": list(opt.band_deg), "action_scale": list(opt.action_scale),
            "built_env": built}


def main_policy(a) -> int:
    pdir = Path(a.policy)
    c = Contract.load()
    meta = json.loads(str(np.load(pdir / "policy.npz")["meta"]))
    if meta.get("kind") != "getup":
        raise SystemExit(f"refused: {pdir} is not a get-up policy (meta kind {meta.get('kind')!r})")
    why = calibration_refusal(meta, c)
    if why:
        raise SystemExit(f"refused: {why}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or pdir.name
    stand = np.asarray(c.stand_dd, dtype=float)
    speeds = [float(v) for v in a.speeds.split(",")]
    summary = {"policy": str(pdir.resolve()), "policy_sha256": _sha(pdir / "policy.npz"), "seed": a.seed,
               "seconds": a.seconds, "tail_s": a.tail_s, "device": a.device,
               "contract_stand_dd": stand.astype(int).tolist(), "end_margin_deg": a.end_margin_deg,
               "period_s": PERIOD, "plant_flags": plant_flags_record(a), "built_env": {}, "files": {}}
    pending = []
    for start in [s for s in a.starts.split(",") if s]:
        r = record_policy(pdir, start, a.seed, a.seconds, a.device, plant_kwargs(a))
        summary["built_env"][start] = r["built_env"]
        n = len(r["dd"]) if r["t_success"] is None else min(len(r["dd"]),
                                                            int(round((r["t_success"] + a.tail_s) / PERIOD)))
        dd = r["dd"][:n]
        diff = np.abs(r["dd_runner"][:n] - dd)
        rec = {"start": start, "success": r["t_success"] is not None, "t_success_s": r["t_success"],
               "recorded_s": n * PERIOD, "filter_alpha": r["filter_alpha"], "limits": r["limits"],
               "band_deg": r["band_deg"], "runner_decoder_max_dd_diff": float(diff.max()),
               "runner_decoder_frames_differing": int((diff > 0).any(-1).sum()),
               "start_pose_dd": r["dd0"].astype(int).tolist(),
               "sim_floor_current_p99_A": np.percentile(r["current_A"][:n], 99, axis=0).round(3).tolist(),
               "sim_floor_current_note": "BAM |tau|/kt of the simulated floor get-up, not of a lifted playback",
               "plant_flags": plant_flags_record(a),
               "built_env": {k: r["built_env"][k] for k in ("sum_body_mass_kg", "masses", "legs", "actuator",
                                                             "attachment", "contract", "limits_file")}}
        print(json.dumps({k: rec[k] for k in ("start", "success", "t_success_s", "recorded_s",
                                               "runner_decoder_max_dd_diff")}), flush=True)
        for sp in speeds:
            delta = resample(dd, sp) - stand[None]
            why = dead_zone_refusal(stand, delta, a.end_margin_deg)
            if why:
                raise SystemExit(f"refused ({start}, speed {sp:g}): {why}; nothing written")
            f = out / f"{name}_{start}_s{a.seed}_{speed_tag(sp)}.npz"
            fmeta = {**rec, "source": "learned get-up policy (getup_stream --policy)", "policy": str(pdir),
                     "policy_sha256": summary["policy_sha256"], "seed": a.seed, "speed": sp, "frames": len(delta),
                     "seconds": len(delta) * PERIOD,
                     "stand_dd_contract": stand.astype(int).tolist(),
                     "end_margin_min_deg": end_margin_deg(stand, delta).round(2).tolist(), **stream_effects(delta)}
            pending.append((f, delta, fmeta))
    for f, delta, fmeta in pending:                     # every start and speed passed the dead-zone check
        np.savez(f, delta_dd=delta, period_s=PERIOD, meta=json.dumps(fmeta))
        summary["files"][f.name] = {k: v for k, v in fmeta.items() if k not in ("start_pose_dd", "stand_dd_contract")}
        print(f"wrote {f} ({fmeta['frames']} frames, {fmeta['seconds']:.2f} s; beyond the stream band "
              f"{fmeta['frames_beyond_stream_band']}, over the per-send step {fmeta['frames_over_stream_step']})",
              flush=True)
    sj = out / f"{name}_s{a.seed}.json"
    sj.write_text(json.dumps(summary, indent=1))
    print("summary", sj)
    return 0


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="a learned get-up's export directory (policy.npz + config.json)")
    ap.add_argument("--starts", default=",".join(POLICY_STARTS), help="getup_eval starts")
    ap.add_argument("--seed", type=int, default=14, help="the env's seed")
    ap.add_argument("--seconds", type=float, default=8.0, help="s simulated per start")
    ap.add_argument("--tail-s", type=float, default=1.0, help="s kept after the get-up's success")
    ap.add_argument("--speeds", default="0.25,1.0", help="speed factors written per start")
    ap.add_argument("--out-dir", default="runs/streams")
    ap.add_argument("--name", help="the files' prefix (default: the policy directory's name)")
    ap.add_argument("--device", default="cpu", help="the env's device (one robot)")
    ap.add_argument("--end-margin-deg", type=float, default=END_MARGIN_DEG,
                    help="refuse a target this close to a track end (default %(default)s, as mp4link stream's)")
    # the plant: each default is the policy's meta's, else its run's options, as getup_eval's
    from mp4_sim.actuators.choice import add_args as add_servo_args

    from mp4_rl.common.bounds import add_legs_args
    from mp4_rl.common.built_env import masses_arg
    add_legs_args(ap, "the policy's meta, else its run's options, else sku_2")
    add_servo_args(ap, evaluation=True)
    ap.add_argument("--masses-json", type=masses_arg, metavar="PATH",
                    help="the plant's masses file (a path, or a file name in the robot's measured data in mp4_sim; default the "
                         "policy's run's)")
    return main_policy(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
