"""Post-install check run by install.sh: no mujoco-core, imports and versions, the mp4-sim release, its robot files, and
the hashes the shipped policies record against mp4-sim's files.

Usage: python verify.py {gpu|cpu}

MP4_SIM_COMMIT (optional): the git commit mp4-sim must have been installed from (compared with the install's record).
"""

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MP4_SIM_VERSION = "0.2.0"
CPU_MODULES = ["numpy", "scipy", "serial", "mujoco", "torch", "mp4_sim", "mp4_rl"]
GPU_MODULES = CPU_MODULES + ["warp", "mujoco_warp", "mjlab", "rsl_rl"]


def sim_source() -> dict:
    """Where pip installed mp4-sim from (PEP 610 direct_url.json): url, requested revision, commit."""
    from importlib import metadata
    try:
        text = metadata.distribution("mp4-sim").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return {}
    d = json.loads(text) if text else {}
    vcs = d.get("vcs_info") or {}
    return {"url": d.get("url"), "requested_revision": vcs.get("requested_revision"), "commit": vcs.get("commit_id")}


def policy_hashes() -> list:
    """For each shipped policy folder: (name, problem or None). The meta's calibration, limits and model hashes and the
    folder's copies against mp4-sim's files."""
    import numpy as np
    from mp4_sim import paths, robots

    out = []
    for folder in sorted((ROOT / "skills").glob("*/policies/sku[12]")):
        r = robots.get(folder.name)
        name = f"{folder.parent.parent.name} {folder.name}"
        if not (folder / "policy.npz").is_file():
            out.append((name, "no policy.npz"))
            continue
        meta = json.loads(str(np.load(folder / "policy.npz", allow_pickle=False)["meta"]))
        bad = []
        if meta.get("contract_sha256") != paths.sha256(r.contract_json):
            bad.append("calibration")
        if ((meta.get("bounds") or {}).get("info") or {}).get("limits_sha256") != paths.sha256(r.limits_json):
            bad.append("joint limits")
        if (meta.get("legs") or {}).get("model_sha256") != paths.sha256(r.xml):
            bad.append("robot model")
        if (meta.get("robot") or "sku2") != r.name:
            bad.append("robot")
        for f in (r.contract_json, r.limits_json):
            if (folder / f.name).read_bytes() != f.read_bytes():
                bad.append(f"the folder's {f.name}")
        out.append((name, ", ".join(bad) or None))
    return out


def mujoco_core() -> str | None:
    """The installed mujoco-core version, if any. mujoco-core (required by mjviser 0.0.17 and newer) installs its own
    mujoco/ package over the pinned mujoco and breaks `import mujoco`; the constraints keep it out."""
    from importlib import metadata

    for name in ("mujoco-core", "mujoco_core"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    modules = GPU_MODULES if mode == "gpu" else CPU_MODULES

    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    failed = []
    core = mujoco_core()
    if core:
        failed.append("mujoco-core")
        print(f"  FAIL mujoco-core {core} is installed: it overwrites the pinned mujoco's files. Delete the virtual "
              "environment and run the installer again (requirements/constraints.txt keeps mujoco-core out).")
    for name in modules:
        try:
            module = importlib.import_module(name)
            print(f"  ok   {name:<20} {getattr(module, '__version__', '')}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL {name:<20} {type(e).__name__}: {e}")

    if "torch" not in failed:
        import torch

        print(f"  torch cuda available: {torch.cuda.is_available()}  (build: {torch.version.cuda})")
        if mode == "gpu" and not torch.cuda.is_available():
            failed.append("torch.cuda")

    if "mp4_sim" not in failed:
        import mp4_sim
        from mp4_sim import robots

        ok = mp4_sim.__version__ == MP4_SIM_VERSION
        print(f"  {'ok  ' if ok else 'FAIL'} mp4_sim version      {mp4_sim.__version__} (expected {MP4_SIM_VERSION})")
        if not ok:
            failed.append("mp4_sim version")
        src = sim_source()
        want = os.environ.get("MP4_SIM_COMMIT")
        print(f"       mp4_sim source       {src.get('url') or 'unknown'} {src.get('requested_revision') or ''} "
              f"{src.get('commit') or ''}".rstrip())
        if want and src.get("commit") != want:
            failed.append("mp4_sim commit")
            print(f"  FAIL mp4_sim commit       {src.get('commit')} (expected {want})")
        for name in ("sku2", "sku1"):
            try:
                n = len(robots.check_files(name))
                print(f"  ok   robot {name}           {n} files match their recorded sha256")
            except Exception as e:
                failed.append(f"robot {name}")
                print(f"  FAIL robot {name}           {e}")
        try:
            for name, why in policy_hashes():
                print(f"  {'ok  ' if why is None else 'FAIL'} policy {name:<13} "
                      f"{'calibration, joint limits and robot model as recorded' if why is None else 'differs: ' + why}")
                if why:
                    failed.append(f"policy {name}")
        except Exception as e:
            failed.append("policies")
            print(f"  FAIL policies            {type(e).__name__}: {e}")

    if mode == "gpu" and "mujoco_warp" not in failed and "mp4_sim" not in failed:
        from mp4_sim.envs.patch_warp import is_patched

        state = is_patched()
        print(f"  mujoco_warp patched: {state}  (python -m mp4_sim.envs.patch_warp)")
        if not state:
            failed.append("mujoco_warp patch")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
