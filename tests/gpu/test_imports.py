"""Every training, evaluation and task module imports (imports alone: no GPU use)."""
import importlib

import pytest

MODULES = ("common.algorithms", "common.anchor_ppo", "common.env", "common.mdp", "common.classic_gait",
           "skills.walker.task", "skills.walker.terrain", "skills.walker.train", "skills.walker.eval.omni", "skills.walker.eval.push",
           "skills.walker.eval.terrain", "skills.walker.eval.gait_metrics", "skills.getup.task", "skills.getup.train", "skills.getup.eval.getup_eval",
           "skills.getup.eval.motion_meter", "skills.getup.eval.sequence")


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(f"mp4_rl.{name}")
