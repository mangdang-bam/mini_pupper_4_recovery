#!/usr/bin/env python3
"""The walker's result tables from the JSONs ``skills/walker/eval/sku2.sh`` writes.

    .venv/bin/python skills/walker/eval/score.py OUT [--markdown FILE]

Omni grid: per command the achieved speed along the command (m/s) or yaw rate (rad/s), the mean of seeds 21 and 22,
against a target of max(0.9 x the classic gait's, 0.75 x the command); a command meets it with every seed at or above
the target and no fall (walk-and-turn commands: both parts). The classic gait is the firmware's trot on the same plant
(``classic_s21/22``). Kicks: falls per kick speed, standing and walking, pooled over the directions and both seeds. Rough
ground: success per terrain kind; stairs: the share that reach the top. The robot counts are read from the JSONs.
Missing JSONs leave their rows out. Needs only the standard library.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

FAMILIES = ("flat", "low_poly", "raised_relief", "slope_up", "slope_down", "boxes", "stairs_up", "stairs_down")
KICK_SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5)


class Results:
    def __init__(self, folder):
        self.folder = Path(folder)

    def __call__(self, name):
        p = self.folder / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def seeds(self, base, seeds):
        """[(seed, data)] of the files present."""
        return [(s, d) for s in seeds if (d := self(f"{base}_s{s}")) is not None]


def parse(key):
    return [float(x) for x in key.split(",")]


def achieved(key, w):
    """(along the command in m/s or None, yaw rate in rad/s or None)."""
    vx, vy, wz = parse(key)
    lin = (vx * vx + vy * vy) ** 0.5
    a_lin = None
    if lin > 0:
        a_lin = w.get("along_mps") if w.get("along_mps") is not None else (w.get("along") or 0.0) * lin
    return a_lin, (w.get("yaw_rate") if wz != 0 else None)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def f(x, fmt="{:.3f}"):
    return "-" if x is None else fmt.format(x)


def omni_table(r, base, classic):
    runs = r.seeds(base, (21, 22))
    if not runs:
        return [], None
    keys = [k for k in runs[0][1]["commands"] if any(parse(k))]
    rows, met = [], 0
    for k in keys:
        vx, vy, wz = parse(k)
        lin = (vx * vx + vy * vy) ** 0.5
        cl = classic.get(k, (None, None))
        targets = [None, None]
        if lin > 0:
            targets[0] = max(0.75 * lin, 0.9 * cl[0] if cl[0] is not None else 0.0)
        if wz != 0:
            sgn = 1.0 if wz > 0 else -1.0
            targets[1] = max(0.75 * abs(wz), 0.9 * sgn * cl[1] if cl[1] is not None else 0.0)
        ok, vals, falls = True, [], 0.0
        for _, d in runs:
            w = d["commands"][k]
            a = achieved(k, w)
            vals.append(a)
            falls = max(falls, w["fall"])
            if w["fall"]:
                ok = False
            if targets[0] is not None and (a[0] is None or a[0] < targets[0] - 1e-9):
                ok = False
            if targets[1] is not None and (a[1] is None or (1.0 if wz > 0 else -1.0) * a[1] < targets[1] - 1e-9):
                ok = False
        met += ok
        got = [mean(v[i] for v in vals) for i in (0, 1)]
        cells = " / ".join(x for x in (f(got[0]) if lin else None, f(got[1], "{:+.2f}") if wz else None) if x)
        cls = " / ".join(x for x in (f(cl[0]) if lin else None, f(cl[1], "{:+.2f}") if wz else None) if x)
        tgt = " / ".join(x for x in (f(targets[0]) if lin else None, f(targets[1], "{:.2f}") if wz else None) if x)
        rows.append(f"| {vx:+.2f} {vy:+.2f} {wz:+.2f} | {cells} | {cls} | {tgt} | "
                    f"{'meets' if ok else 'misses'}{f' (falls {100 * falls:.1f} %)' if falls else ''} |")
    head = ["| vx m/s, vy m/s, wz rad/s | achieved | classic gait | target | |", "|---|---|---|---|---|"]
    return head + rows, (met, len(keys), [s for s, _ in runs])


def classic_values(r):
    out = {}
    runs = r.seeds("classic", (21, 22))
    if not runs:
        return out, []
    for k in runs[0][1]["commands"]:
        a = [achieved(k, d["commands"][k]) for _, d in runs]
        out[k] = (mean(x[0] for x in a), mean(x[1] for x in a))
    return out, [s for s, _ in runs]


def push_table(r):
    runs = r.seeds("push", (21, 22))
    if not runs:
        return []
    rows = ["| kick speed | falls standing | falls walking |", "|---|---|---|"]
    for mag in KICK_SPEEDS:
        cells = []
        for standing in (True, False):
            n = nf = 0
            for _, d in runs:
                for c in d["cells"].values():
                    if c["kind"] == "kick" and abs(c["mag"] - mag) < 1e-9 and (c["command"] == "stand") == standing:
                        n, nf = n + c["n"], nf + c["n_fall"]
            cells.append(f"{100 * nf / n:.1f} % of {n}" if n else "-")
        rows.append(f"| {mag:.2f} m/s | " + " | ".join(cells) + " |")
    return rows


def terrain_rows(r):
    rows = []
    for name, label in (("wrough_s21", "rough ground"), ("wrough_lidar_s21", "rough ground, lidar")):
        d = r(name)
        if d is None:
            continue
        t = d["trials"]
        n = len(t["success"])
        parts = []
        for fam in FAMILIES:
            idx = [i for i in range(n) if t["terrain"][i] == fam]
            if idx:
                parts.append(f"{fam} {100 * sum(t['success'][i] for i in idx) / len(idx):.1f}")
        rows.append(f"| {label} | {100 * sum(t['success']) / n:.1f} % of {n} ({', '.join(parts)}) |")
    for name, label in (("wstairs_s21", "four 25 mm stairs"), ("wstairs_lidar_s21", "four 25 mm stairs, lidar")):
        d = r(name)
        if d is None:
            continue
        t = d["trials"]
        n = len(t["success"])
        rows.append(f"| {label} | {100 * sum(t['success']) / n:.1f} % of {n} reach the top, "
                    f"{100 * sum(t['fall']) / n:.1f} % fall |")
    return (["| test (seed 21) | success |", "|---|---|"] + rows) if rows else []


def report(folder) -> str:
    r = Results(folder)
    out = [f"# The walker in simulation ({Path(folder).name})", ""]
    classic, cseeds = classic_values(r)
    for base, label in (("omni", "without the lidar"), ("omni_lidar", "with the lidar")):
        rows, summary = omni_table(r, base, classic)
        if not rows:
            continue
        met, n, seeds = summary
        per = r.seeds(base, (21, 22))[0][1].get("per")
        out += [f"## The walker's commands, {label}", "",
                f"{met} of {n} commands meet their target ({per} robots per command; mean of seeds "
                f"{', '.join(map(str, seeds))}; classic gait seeds {', '.join(map(str, cseeds)) or 'none'}).", ""]
        out += rows + [""]
    rows = push_table(r)
    if rows:
        out += ["## Kicks", "", "Falls per kick, eight directions, standing and while walking (seeds 21, 22):", ""]
        out += rows + [""]
    rows = terrain_rows(r)
    if rows:
        out += ["## Rough ground and stairs", ""] + rows + [""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the folder skills/walker/eval/sku2.sh wrote")
    ap.add_argument("--markdown", metavar="FILE", help="also write the tables to FILE")
    a = ap.parse_args(argv)
    if not Path(a.folder).is_dir():
        print(f"no folder {a.folder}", file=sys.stderr)
        return 2
    text = report(a.folder)
    print(text)
    if a.markdown:
        Path(a.markdown).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
