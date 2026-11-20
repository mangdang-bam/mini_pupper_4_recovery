#!/usr/bin/env python3
"""Install a joint-limits file (and a hip envelope) as the robot's, recording the limits file's hash.

    .venv/bin/python tools/install_limits.py --limits fit/sku2_joint_limits.json [--hip-envelope fit/sku2_hip_envelope.json]
    .venv/bin/python tools/install_limits.py --robot sku1 --limits fit/sku1_joint_limits.json --hip-envelope fit/sku1_hip_envelope.json

The files are checked first (four legs with their ranges and signs, an optional ``thigh_stop_deg_backward`` per leg, the
decoder's bounds buildable in both modes under the robot's calibration, the envelope's format), then copied byte for
byte into the installed ``mp4_sim``'s measured data of the robot (``--robot``, default sku2; ``--data``: another folder)
under the robot's file names, and the limits file's sha256 written to ``<robot>_joint_limits.sha256``, the one place
``mp4_sim.limits.load`` checks it against. Install ``mp4_sim`` from a source checkout in editable mode first
(docs/INSTALL.md, "Another unit's joint limits"), so the files land in the checkout rather than in ``site-packages``.
``mp4_sim.robots.check_files`` (``scripts/install/verify.py``) then reports the replaced files until their hashes in
the checkout's ``src/mp4_sim/robots.py`` are updated too. Every policy's decoder
bounds come from these files: re-export each with ``python -m mp4_rl.common.export --from-policy DIR DIR`` (which also
copies the new limits into the policy's folder) and check it with ``runner check`` afterwards.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from mp4_rl import paths
from mp4_rl.common import bounds as limits

LEG_KEYS = ("abd_out_range", "knee_fold_range", "signs_dd_to_phys")


def check_limits(j: dict, robot=None) -> None:
    """Refuse a limits file the code cannot use."""
    legs = [d.get("leg") for d in j.get("legs", [])]
    if sorted(legs) != sorted(limits.LEGS):
        raise ValueError(f"legs {legs}: one entry for each of {list(limits.LEGS)} expected")
    for d in j["legs"]:
        miss = [k for k in LEG_KEYS if d.get(k) is None]
        if miss:
            raise ValueError(f"{d['leg']}: missing {miss}")
        if set(d["signs_dd_to_phys"]) < {"abd_out", "hip_fwd", "knee_fold"}:
            raise ValueError(f"{d['leg']}: signs_dd_to_phys needs abd_out, hip_fwd, knee_fold")
        stop = d.get(limits.THIGH_STOP_KEY)
        if stop is not None and not (isinstance(stop, (int, float)) and 0.0 < float(stop) < 360.0):
            raise ValueError(f"{d['leg']}: {limits.THIGH_STOP_KEY} {stop!r}: a number of degrees (0..360) or null")
    for mode in limits.MODES:
        limits.decoder_bounds(mode, j=j, envelope=False, robot=robot)


def check_envelope(path: Path, j: dict, robot=None) -> None:
    env = limits.hip_envelope(path=path, robot=robot)
    if not isinstance(env, dict):
        raise ValueError(f"{path}: no 'envelope' object")
    b = limits.decoder_bounds("getup", j=j, envelope=False, robot=robot)
    dataclasses.replace(b, hip_envelope=env).check()


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def install(limits_path, hip_envelope=None, data_dir=None, robot=None) -> str:
    """Check and copy; returns the recorded sha256."""
    from mp4_sim import robots
    r = robots.get(robot)
    data_dir = Path(data_dir) if data_dir is not None else r.measured
    src = Path(limits_path)
    raw = src.read_bytes()
    j = json.loads(raw)
    check_limits(j, r)
    env_raw = None
    if hip_envelope is not None:
        check_envelope(Path(hip_envelope), j, r)
        env_raw = Path(hip_envelope).read_bytes()
    if data_dir == r.measured and "site-packages" in data_dir.parts:
        print(f"note: {data_dir} is inside an installed mp4_sim, not a source checkout; see docs/INSTALL.md "
              "(\"Another unit's joint limits\")", file=sys.stderr)
    dst = data_dir / r.limits_json.name
    sha_file = data_dir / r.limits_sha256_file.name
    old = sha_file.read_text().split()[0] if sha_file.exists() else None
    _write(dst, raw)
    if env_raw is not None:
        _write(data_dir / r.hip_envelope_json.name, env_raw)
    sha = paths.sha256(dst)
    _write(sha_file, f"{sha}  {dst.name}\n".encode())
    stops = {d["leg"]: d.get(limits.THIGH_STOP_KEY) for d in j["legs"] if d.get(limits.THIGH_STOP_KEY) is not None}
    print(f"installed {src} -> {dst}" + (f" and {hip_envelope} -> {data_dir / r.hip_envelope_json.name}"
                                         if env_raw is not None else ""))
    print(f"sha256 {old or '-'} -> {sha} ({sha_file.name}); thigh stops (deg backward): {stops or 'none'}")
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limits", required=True, help="a joint-limits file (the robot's format, e.g. sku2_joint_limits.json)")
    ap.add_argument("--hip-envelope", help="a hip envelope file (e.g. sku2_hip_envelope.json)")
    ap.add_argument("--data", help="the folder to install into (default: the installed mp4_sim's data of the robot)")
    from mp4_sim import robots
    robots.add_args(ap)
    a = ap.parse_args(argv)
    try:
        install(a.limits, a.hip_envelope, a.data, a.robot)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print("next: re-export every policy's bounds (.venv/bin/python -m mp4_rl.common.export --from-policy DIR DIR) and, for the "
          "SKU2's, run runner check on each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
