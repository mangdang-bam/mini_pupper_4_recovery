#!/usr/bin/env python3
"""The get-up's result tables from the JSONs ``mp4_rl/skills/getup/eval/sku2.sh`` writes.

    .venv/bin/python mp4_rl/skills/getup/eval/score.py OUT [--markdown FILE]

Success per start, without and with the lidar, per seed; the sequence's walking-again share within 8 s of a fall, the
robots that never got up, and the times; get-ups on rough ground and stairs. The robot counts are read from the JSONs.
Missing JSONs leave their rows out. Needs only the standard library.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

STARTS = ("left", "right", "back", "nose", "midfall", "random")
GETUP_TERRAIN = ("low_poly", "raised_relief", "slope_up", "slope_down", "boxes")


def seq_robots(d: dict):
    """The sequence's robot count: its JSON keeps the per-robot results under ``robots``."""
    n = d.get("robots")
    return len(n["kicks"]) if isinstance(n, dict) and "kicks" in n else n


class Results:
    def __init__(self, folder):
        self.folder = Path(folder)

    def __call__(self, name):
        p = self.folder / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def seeds(self, base, seeds):
        """[(seed, data)] of the files present."""
        return [(s, d) for s in seeds if (d := self(f"{base}_s{s}")) is not None]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def f(x, fmt="{:.3f}"):
    return "-" if x is None else fmt.format(x)


def getup_tables(r):
    out = []
    for base, label in (("gflat", "flat ground"), ("glidar", "flat ground, lidar")):
        runs = r.seeds(base, (21, 22, 23, 24))
        if not runs:
            continue
        n = runs[0][1].get("robots")
        out += [f"Get-up success from each start, {label} ({n} robots per start; seeds "
                f"{', '.join(str(s) for s, _ in runs)}):",
                "", "| start | " + " | ".join(f"seed {s}" for s, _ in runs) + " | time to up, p50 |",
                "|---|" + "---|" * (len(runs) + 1)]
        for st in STARTS:
            vals = [d["starts"][st]["summary"] for _, d in runs if st in d["starts"]]
            if vals:
                out.append(f"| {st} | " + " | ".join(f"{100 * v['success']:.1f} %" for v in vals)
                           + f" | {f(mean(v.get('t_success_p50') for v in vals), '{:.2f} s')} |")
        out.append("")
    runs = r.seeds("seq", (21, 22, 23, 24))
    if runs:
        d0 = runs[0][1]
        out += [f"The sequence ({seq_robots(d0)} robots, {d0.get('seconds'):g} s, with the walker; seeds "
                f"{', '.join(str(s) for s, _ in runs)}):", "",
                "| | " + " | ".join(f"seed {s}" for s, _ in runs) + " |", "|---|" + "---|" * len(runs)]
        S = [d["summary"] for _, d in runs]
        out.append("| walking again within 8 s of a fall | " + " | ".join(f"{100 * s['overall']['within_8s']:.1f} %" for s in S) + " |")
        out.append("| falls | " + " | ".join(str(s["overall"]["toppled"]) for s in S) + " |")
        out.append("| robots that never got up | " + " | ".join(str(s["never_up"]) for s in S) + " |")
        out.append("| back to walking, median / 90th percentile | "
                   + " | ".join(f"{s['overall']['t_walk_s_p50']:.1f} / {s['overall']['t_walk_s_p90']:.1f} s" for s in S) + " |")
        out.append("")
    rows, n = [], None
    for s in (21, 23):
        d = r(f"gterr_s{s}")
        if d is not None:
            n = d.get("robots")
            rows.append(f"| rough ground, seed {s} | " + ", ".join(
                f"{fam} {100 * d['by_family'][fam]['all']['up']:.1f}" for fam in GETUP_TERRAIN if fam in d["by_family"]) + " |")
        d = r(f"gstairs_s{s}")
        if d is not None:
            n = n or d.get("robots")
            rows.append(f"| stairs, seed {s} | " + ", ".join(
                f"{fam} {100 * d['by_family'][fam]['all']['up']:.1f} (from the back {100 * d['starts']['back']['by_family'][fam]['up']:.1f})"
                for fam in ("stairs4", "stairs8") if fam in d["by_family"]) + " |")
    if rows:
        out += [f"Get-ups on rough ground and stairs ({n} robots over the starts; % up):", "",
                "| test | up, % |", "|---|---|"] + rows + [""]
    return out


def report(folder) -> str:
    r = Results(folder)
    out = [f"# The get-up in simulation ({Path(folder).name})", ""]
    rows = getup_tables(r)
    if rows:
        out += ["## The get-up", ""] + rows
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the folder mp4_rl/skills/getup/eval/sku2.sh wrote")
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
