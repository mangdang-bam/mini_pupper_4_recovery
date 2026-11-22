#!/usr/bin/env python3
"""Summarise robot runs from their logs: per run, and the totals.

    python3 tools/summarise_runs.py logs/dir1 logs/dir2 ...

Each run is a ``<name>.policy.npz`` with its ``<name>.npz``. Numpy only.

A **placed** get-up is the first of a ``--start-mode getup`` run; any other get-up is a **fall**. A get-up is **up** if it handed
back, and up **on the first attempt** if its phase clock never restarted. A get-up still running when the run ended, whose last
attempt had less than the phase's length plus its restart delay, is **cut off** and judged neither way.
An **attempt** runs from the get-up's entry or a restart to the next restart or the get-up's end; its **max tilt** is the highest
filtered tilt in it. Falls are sided by the filtered gravity at the entry: **right** flank lower below y -0.1, **left** above 0.1.
``--getup-policy NAME`` keeps the runs whose get-up policy's folder is NAME; ``--attempts`` prints every attempt.
Temperatures: the housing's peak, a 3-frame median over each servo taken first, since single frames can read double (``raw``
is the peak without it).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

OVER_DEG = 170.0
SIDE_GRAVITY_Y = 0.1


def median3(a: np.ndarray) -> np.ndarray:
    if len(a) < 3:
        return a
    return np.median(np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)


def summarise(base: Path) -> dict:
    z = np.load(f"{base}.policy.npz")
    m = json.loads(str(z["meta"]))
    r = np.load(f"{base}.npz")
    temp = r["temp_dc"].astype(float) / 10.0
    cur = np.abs(r["cur_ma"].astype(float)) / 1000.0
    out = {"name": base.name, "getup_policy": Path(m.get("getup_policy", "")).name, "steps": len(z["tick"]),
           "cmd": (m.get("vx"), m.get("vy"), m.get("wz")),
           "start_mode": m.get("start_mode", "walk"), "seconds": m.get("seconds"), "stop": m.get("stop_reason", ""),
           "peak_A": float(cur.max()) if cur.size else 0.0, "temp_start": float(temp[0].max()) if len(temp) else 0.0,
           "temp_peak": float(median3(temp).max()) if len(temp) else 0.0, "temp_raw": float(temp.max()) if len(temp) else 0.0,
           "tilt_max": float(np.nanmax(z["tilt_cf_deg"])) if len(z["tick"]) else 0.0,
           "reading_flags": len(m.get("reading_flags") or []),
           "first_step_gap_ms": first_step_gap_ms(z, r), "events": []}
    phase = float(m.get("getup_phase_s", 4.0) or 4.0) + float(m.get("getup_phase_restart_s", 1.0) or 1.0)
    last_tick = int(z["tick"][-1]) if len(z["tick"]) else 0
    for i, e in enumerate(m.get("getups") or []):
        placed = i == 0 and e.get("trigger") == "start mode getup"
        restarts = e.get("phase_restarts") or []
        up = e.get("end") == "hand-back"
        t = e.get("seconds") if up else ((last_tick - int(e["tick"])) & 0xFFFFFFFF) * 0.005
        last_start = int(z["tick"][restarts[-1]]) if restarts else int(e["tick"])
        last_t = ((last_tick - last_start) & 0xFFFFFFFF) * 0.005
        out["events"].append({"placed": placed, "up": up, "first": up and not restarts, "seconds": t, "restarts": len(restarts),
                              "cut_off": (not up) and e.get("end") is None and last_t < phase, "end": e.get("end"),
                              "side": side_of(e.get("gravity")), "attempts": attempts(e, z)})
    return out


def side_of(gravity) -> str:
    """Which flank was lower at a get-up's entry, from the filtered gravity (body frame, y to the left)."""
    if gravity is None or abs(gravity[1]) <= SIDE_GRAVITY_Y:
        return "neither"
    return "right" if gravity[1] < 0 else "left"


def attempts(e: dict, z) -> list:
    """Per attempt of get-up ``e``: its first step, length and max tilt."""
    n = len(z["tick"])
    end = n if e.get("end_step") is None else min(int(e["end_step"]), n)
    edges = [int(e["step"]), *(int(k) for k in e.get("phase_restarts") or []), end]
    tilt = z["tilt_cf_deg"]
    return [{"step": a, "seconds": (b - a) * 0.02, "max_tilt_deg": float(np.nanmax(tilt[a:max(b, a + 1)]))}
            for a, b in zip(edges[:-1], edges[1:]) if a < n]


def first_step_gap_ms(z, r):
    """The gap in sends across the first policy step, or None for a run that started with it."""
    st = r["sent_t"]
    if not len(z["t_send"]):
        return float((st[-2] - st[-3]) * 1e3) if len(st) >= 3 else None      # it never logged a step: the send before the disarm
    i = int(np.searchsorted(st, z["t_send"][0]))
    return float((st[i] - st[i - 1]) * 1e3) if i > 0 else None


def totals(rows: list) -> dict:
    t = {k: {"n": 0, "up": 0, "first": 0, "not_up": 0, "cut_off": 0, "restarts": 0, "times": []} for k in ("placed", "fall")}
    for r in rows:
        for e in r["events"]:
            b = t["placed" if e["placed"] else "fall"]
            b["n"] += 1
            b["up"] += e["up"]
            b["first"] += e["first"]
            b["cut_off"] += e["cut_off"]
            b["not_up"] += not e["up"] and not e["cut_off"]
            b["restarts"] += e["restarts"]
            if e["up"]:
                b["times"].append(e["seconds"])
    return t


def first_attempts_over(rows: list) -> dict:
    """Falls' first attempts past ``OVER_DEG``: overall and per side, each ``{"n", "over"}``."""
    t = {k: {"n": 0, "over": 0} for k in ("all", "right", "left", "neither")}
    for r in rows:
        for e in r["events"]:
            if e["placed"] or not e["attempts"]:
                continue
            for k in ("all", e["side"]):
                t[k]["n"] += 1
                t[k]["over"] += e["attempts"][0]["max_tilt_deg"] > OVER_DEG
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", help="also write everything to this file")
    ap.add_argument("--getup-policy", metavar="NAME", help="only the runs with this get-up policy (its folder's name)")
    ap.add_argument("--attempts", action="store_true", help="print every attempt of every get-up")
    a = ap.parse_args(argv)
    rows = []
    for d in a.dirs:
        rows += [summarise(Path(str(f)[:-len(".policy.npz")])) for f in sorted(Path(d).glob("*.policy.npz"))]
    if a.getup_policy:
        rows = [r for r in rows if r["getup_policy"] == a.getup_policy]
    if not rows:
        print("no logs found", file=sys.stderr)
        return 1
    for r in rows:
        ev = r["events"]
        times = [round(e["seconds"], 2) for e in ev if e["up"]]
        gap = "-" if r["first_step_gap_ms"] is None else f"{r['first_step_gap_ms']:.0f}"
        print(f"{r['name']:12s} steps {r['steps']:4d} cmd {r['cmd']} {r['start_mode']:5s} get-ups {len(ev)} up {sum(e['up'] for e in ev)} "
              f"first {sum(e['first'] for e in ev)} cut off {sum(e['cut_off'] for e in ev)} restarts {[e['restarts'] for e in ev]} "
              f"t {times} peak {r['peak_A']:.2f} A T {r['temp_start']:.1f}->{r['temp_peak']:.1f} C (raw {r['temp_raw']:.1f}) "
              f"tilt max {r['tilt_max']:.0f} near-end readings {r['reading_flags']} first-step gap {gap} ms; {r['stop']}")
        for i, e in enumerate(ev if a.attempts else []):
            kind = "placed" if e["placed"] else "fall"
            print(f"  {kind} {i + 1} {e['side']} {e['end'] or ('cut off' if e['cut_off'] else 'no end')}: " + ", ".join(
                f"{at['seconds']:.2f} s to {at['max_tilt_deg']:.0f} deg" for at in e["attempts"]))
    t = totals(rows)
    for k, b in t.items():
        span = f"{min(b['times']):.2f}-{max(b['times']):.2f} s, median {np.median(b['times']):.2f} s" if b["times"] else "-"
        print(f"{k}: {b['n']} get-ups, {b['up']} up, {b['first']} on the first attempt, {b['not_up']} never up, {b['cut_off']} cut "
              f"off by the run's end, {b['restarts']} restarts; time to hand-back {span}")
    over = first_attempts_over(rows)
    print(f"falls' first attempts over {OVER_DEG:.0f} deg: " + ", ".join(
        f"{k} {v['over']} of {v['n']}" for k, v in over.items()))
    gaps = [r["first_step_gap_ms"] for r in rows if r["first_step_gap_ms"] is not None]
    if gaps:
        print(f"first-step gap ms: {sorted(round(g) for g in gaps)}")
    print(f"peak current {max(r['peak_A'] for r in rows):.2f} A; housing peak {max(r['temp_peak'] for r in rows):.1f} C "
          f"(raw {max(r['temp_raw'] for r in rows):.1f} C)")
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": rows, "totals": t, "first_attempts_over": over}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
