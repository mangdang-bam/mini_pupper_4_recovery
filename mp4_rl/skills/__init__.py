"""The bank of policies: one self-contained folder per policy (its README, code, recipes, policies, checkpoints and
evaluation), registered here in ``REGISTRY``. Adding a policy: docs/ADDING_A_POLICY.md.
Import-free, so the robot side can read it."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBOTS = ("sku2", "sku1")

# name: what the policy does
REGISTRY = {
    "walker": "walks in every direction, turns in place and rides out shoves",
    "getup": "gets the robot back on its feet from any fall pose and hands back to the walker",
}


def folder(name: str) -> Path:
    """The policy's folder, ``mp4_rl/skills/<name>``."""
    if name not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}: one of {sorted(REGISTRY)}")
    return HERE / name


def policy_folder(name: str, robot: str) -> Path:
    """The exported policy for one robot, ``mp4_rl/skills/<name>/policies/<robot>``."""
    if robot not in ROBOTS:
        raise KeyError(f"unknown robot {robot!r}: one of {ROBOTS}")
    return folder(name) / "policies" / robot


def shipped() -> list[tuple[str, str, Path]]:
    """Every registered policy's exported folders that exist: (name, robot, folder)."""
    return [(n, r, policy_folder(n, r)) for n in REGISTRY for r in ROBOTS if policy_folder(n, r).is_dir()]
