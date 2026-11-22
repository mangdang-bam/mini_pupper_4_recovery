"""Shared by the robot-side tests: synthetic policies and a run against the fake robot."""
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fake_esp32 import FakeRobot

from mp4_rl import paths
from mp4_rl.common import policy as cmod
from mp4_rl.common import policy_file
from mp4_rl.common.policy import CONTRACT_JSON, Contract
from mp4_rl.deploy.runner import checks
from mp4_rl.deploy.runner import runner as rn

# A fixed TEST calibration (offsets -12, 8, -16, 6, -36, -42, -8, 25, 4, 17, 7, 19; not the packaged contract's). The fake
# robot, the runner, the link and the tools are tested against it, so that a new calibration of the real robot (a new
# packaged contract) doesn't change what these tests check. Tests of the
# packaged contract and policies use the packaged contract (``PACKAGED``, ``CONTRACT_JSON``).
TEST_CONTRACT_JSON = Path(__file__).resolve().parent / "data" / "sku2_contract_test.json"
C = Contract.load(TEST_CONTRACT_JSON)
PACKAGED = Contract.load(CONTRACT_JSON)
MOUNT = rn.load_mount()[1]
# What a policy trained for this robot names in its meta (the runner refuses others: ``checks.robot_refusal``).
ROBOT_META = {"actuator": checks.ROBOT_ACTUATOR, "legs": dict(checks.ROBOT_LEGS)}
RAMP_TICKS = 40                 # armed ticks at the policy's first step: the 0.2 s ramp these tests use


def make_policy(d, bias=0.0, scale=(0.10, 0.40, 0.42), calibration=TEST_CONTRACT_JSON, **meta_extra):
    """A one-layer policy whose action is ``bias`` (a scalar or 12 values) whatever it observes, exported for
    ``calibration`` (copied into its folder as ``sku2_contract.json``; None: no calibration in the folder)."""
    d.mkdir(parents=True)
    if calibration is not None:
        shutil.copyfile(calibration, d / "sku2_contract.json")
    sha = paths.sha256(calibration if calibration is not None else TEST_CONTRACT_JSON)
    meta = {"contract_sha256": sha, "omni": True, **ROBOT_META, **meta_extra}
    b = np.broadcast_to(np.asarray(bias, dtype=np.float32), (12,)).copy()
    s = np.broadcast_to(np.asarray(scale * 4 if len(scale) == 3 else scale, dtype=np.float32), (12,)).copy()
    np.savez(d / "policy.npz", n_layers=1, scale=s, meta=json.dumps(meta), fixture_obs=np.zeros((2, 45), dtype=np.float32),
             fixture_act=np.tile(b, (2, 1)), W0=np.zeros((12, 45), dtype=np.float32), b0=b)
    return d


def getup_policy(d, bias=0.0, **meta):
    """A get-up policy: kind getup, the 90 deg band, 1.57 rad per unit, a 4 s phase clock."""
    m = {"kind": "getup", "band_deg": [90.0] * 12, "getup_phase_s": 4.0, **meta}
    return make_policy(d, bias, scale=(1.57,) * 12, **m)


def use_test_contract(monkeypatch, path=TEST_CONTRACT_JSON):
    """Make ``path`` the calibration the code under test loads by default: ``Contract.load()``'s default and every
    module's ``CONTRACT_JSON`` (the runner itself reads each policy folder's calibration)."""
    path = Path(path)
    monkeypatch.setattr(Contract.load.__func__, "__defaults__", (path, None))
    for mod in (cmod, policy_file):
        monkeypatch.setattr(mod, "CONTRACT_JSON", path)


@pytest.fixture
def fixed_calibration(monkeypatch):
    """The code under test runs at the fixed test calibration (``C``)."""
    use_test_contract(monkeypatch)
    return C


def fake_robot(monkeypatch, contract):
    gc.collect()   # the fake shares this process: a collection of earlier tests' garbage would stall its STATE
    monkeypatch.setattr(rn, "HOLD_AFTER_S", 0.2)
    return FakeRobot(contract, MOUNT, rn.IDENTIFIED)


@pytest.fixture
def robot(monkeypatch, fixed_calibration):
    """The fake robot at the test calibration."""
    f = fake_robot(monkeypatch, C)
    yield f
    f.close()


@pytest.fixture
def packaged_robot(monkeypatch):
    """The fake robot at the packaged contract's calibration (for the packaged policies)."""
    f = fake_robot(monkeypatch, PACKAGED)
    yield f
    f.close()


def go(robot, tmp_path, *extra, seconds=3.0, walker_bias=0.0, getup_bias=0.0, walker_meta=None, getup_meta=None):
    """One ``run`` with synthetic policies; returns (exit code, the step log, its meta)."""
    walker = make_policy(tmp_path / "walker", walker_bias, **(walker_meta or {}))
    getup = getup_policy(tmp_path / "getup", getup_bias, **(getup_meta or {}))
    log = tmp_path / "run"
    rc = rn.main(["--port", robot.port, "run", "--policy", str(walker), "--getup-policy", str(getup), "--stand-ramp", "0.2",
                  "--seconds", str(seconds), "--log", str(log), *extra])
    if rc:
        return rc, None, None
    z = np.load(log.with_suffix(".policy.npz"))
    return rc, z, json.loads(str(z["meta"]))
