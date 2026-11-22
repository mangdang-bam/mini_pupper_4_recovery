"""Every module the CPU install needs imports without a GPU or the GPU packages."""
import importlib

import pytest

MODULES = ("paths", "common.policy", "common.bounds", "common.heading", "common.obs_norm", "common.policy_file",
           "common.export", "common.actor", "common.built_env", "skills.getup.eval.reach", "deploy.runner.runner",
           "deploy.runner.checks", "deploy.mp4link.cli", "skills")


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(f"mp4_rl.{name}")
