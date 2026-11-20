"""Every training, evaluation and task module imports (imports alone: no GPU use)."""
import importlib

import pytest

MODULES = ("common.algorithms", "common.anchor_ppo", "common.env", "common.mdp", "common.classic_gait",
           "walker.task", "walker.terrain", "walker.train", "walker.eval.omni", "walker.eval.push",
           "walker.eval.terrain", "walker.eval.gait_metrics", "getup.task", "getup.train", "getup.eval.getup_eval",
           "getup.eval.motion_meter", "getup.eval.sequence")


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(f"mp4_rl.{name}")
