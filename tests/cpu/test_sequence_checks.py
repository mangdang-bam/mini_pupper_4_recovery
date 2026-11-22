"""The sequence evaluation's checks before it builds anything: each policy against the evaluated robot's calibration."""
import json
import types

import numpy as np
import pytest

from mp4_rl import skills
from mp4_rl.skills.getup.eval import sequence as sq


def metas(*folders):
    return [json.loads(str(np.load(f / "policy.npz")["meta"])) for f in folders]


@pytest.mark.parametrize("robot", skills.ROBOTS)
def test_the_shipped_pair_of_each_robot_is_accepted(robot):
    dirs = [skills.policy_folder("walker", robot), skills.policy_folder("getup", robot)]
    a = types.SimpleNamespace(robot=None)
    assert sq.eval_robot(a, metas(*dirs)) == robot
    assert sq.calibration_refusal(a, dirs, metas(*dirs)) is None


def test_a_pair_for_different_robots_is_refused(capsys):
    w, g = skills.policy_folder("walker", "sku1"), skills.policy_folder("getup", "sku2")
    rc = sq.main(["--walker", str(w), "--getup", str(g), "--robots", "4", "--seconds", "1", "--json", "unused.json"])
    assert rc == 2 and "exported for different robots (sku1 / sku2)" in capsys.readouterr().out


def test_a_policy_for_another_calibration_is_refused():
    dirs = [skills.policy_folder("walker", "sku2"), skills.policy_folder("getup", "sku2")]
    why = sq.calibration_refusal(types.SimpleNamespace(robot="sku1"), dirs, metas(*dirs))
    assert why and why.startswith(f"{dirs[0]}: ") and "calibration" in why
