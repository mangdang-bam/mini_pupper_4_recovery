#!/usr/bin/env python3
"""Install a joint-limits file (and a hip envelope) as the packaged ones, recording the limits file's hash.

    python tools/install_limits.py --limits fit/sku2_joint_limits.json [--hip-envelope fit/sku2_hip_envelope.json]

The files are checked first (four legs with their ranges and signs, an optional ``thigh_stop_deg_backward`` per leg, the
decoder's bounds buildable in both modes under the packaged contract, the envelope's format), then copied byte for byte
into ``src/mp4_recovery/data`` (``--data``: another folder), and the limits file's sha256 written to
``sku2_joint_limits.sha256``, the one place ``limits.load`` checks it against. Every policy's decoder bounds come from
these files: re-export each with ``python -m mp4_recovery.export_policy --from-policy DIR DIR`` and check it with
``runner check`` afterwards.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from mp4_recovery import limits, paths

LEG_KEYS = ("abd_out_range", "knee_fold_range", "signs_dd_to_phys")


def check_limits(j: dict) -> None:
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
        limits.decoder_bounds(mode, j=j, envelope=False)


def check_envelope(path: Path, j: dict) -> None:
    env = limits.hip_envelope(path=path)
    if not isinstance(env, dict):
        raise ValueError(f"{path}: no 'envelope' object")
    b = limits.decoder_bounds("getup", j=j, envelope=False)
    dataclasses.replace(b, hip_envelope=env).check()


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def install(limits_path, hip_envelope=None, data_dir=None) -> str:
    """Check and copy; returns the recorded sha256."""
    data_dir = Path(data_dir) if data_dir is not None else paths.DATA
    src = Path(limits_path)
    raw = src.read_bytes()
    j = json.loads(raw)
    check_limits(j)
    env_raw = None
    if hip_envelope is not None:
        check_envelope(Path(hip_envelope), j)
        env_raw = Path(hip_envelope).read_bytes()
    dst = data_dir / limits.LIMITS_JSON.name
    sha_file = data_dir / limits.LIMITS_SHA256_FILE.name
    old = sha_file.read_text().split()[0] if sha_file.exists() else None
    _write(dst, raw)
    if env_raw is not None:
        _write(data_dir / limits.HIP_ENVELOPE_JSON.name, env_raw)
    sha = paths.sha256(dst)
    _write(sha_file, f"{sha}  {dst.name}\n".encode())
    stops = {d["leg"]: d.get(limits.THIGH_STOP_KEY) for d in j["legs"] if d.get(limits.THIGH_STOP_KEY) is not None}
    print(f"installed {src} -> {dst}" + (f" and {hip_envelope} -> {data_dir / limits.HIP_ENVELOPE_JSON.name}"
                                         if env_raw is not None else ""))
    print(f"sha256 {old or '-'} -> {sha} ({sha_file.name}); thigh stops (deg backward): {stops or 'none'}")
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limits", required=True, help="a sku2_joint_limits.json")
    ap.add_argument("--hip-envelope", help="a sku2_hip_envelope.json")
    ap.add_argument("--data", help="the folder to install into (default: the package's data folder)")
    a = ap.parse_args(argv)
    try:
        install(a.limits, a.hip_envelope, a.data)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print("next: re-export every policy's bounds (python -m mp4_recovery.export_policy --from-policy DIR DIR) and run "
          "runner check on each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
