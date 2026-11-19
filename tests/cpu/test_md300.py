"""md300: the vendored MD-300-150 model and its gains, checked on the CPU against BAM's own class and the deploy code."""
import importlib
import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mp4_recovery import paths
from mp4_recovery.actuators import choice
from mp4_recovery.actuators.bam.model import load_model

# a checkout of Rhoban/bam with the MD-300 model (pull request #26, 2d1688a): compared against when BAM_UPSTREAM names it
UPSTREAM = Path(os.environ["BAM_UPSTREAM"]) if os.environ.get("BAM_UPSTREAM") else None
SETS = {1: "md300_id_1_m6.json", 2: "md300_id_2_m6.json", 3: "md300_id_3_m6.json"}
# operating points: (position error rad, velocity rad/s); the first two inside / at the 0.1 deg dead band
POINTS = [(math.radians(0.05), 0.0), (math.radians(0.1), 0.0), (math.radians(0.5), 0.0), (math.radians(2.0), 1.5),
          (math.radians(-5.0), -3.0), (math.radians(20.0), 0.0), (math.radians(60.0), 4.0), (math.radians(-60.0), 6.0),
          (0.0, 2.0), (math.radians(1.0), -8.0)]


def test_robot_gains_match_the_deploy_code():
    from mp4_recovery.deploy import runner
    from mp4_recovery.deploy.mp4link.cli import GAINS
    assert GAINS["control_mode"] == 1.0
    assert (GAINS["kp_position"], GAINS["kd_position"], GAINS["kff_current"], GAINS["max_pwm_duty_cycle"]) == \
        (choice.ROBOT_KP, choice.ROBOT_KD, choice.ROBOT_KFF, choice.ROBOT_MAX_PWM)
    assert runner.CAP_MA == choice.ROBOT_CAP_MA
    fw = choice.MD300_FIRMWARE
    assert fw["pwm_scale"] == GAINS["kff_current"] and fw["kp"] == GAINS["kp_position"] and fw["kd"] == GAINS["kd_position"]
    assert choice.duty_per_ma("md01") == choice.duty_per_ma("md300") == 0.001


@pytest.mark.parametrize("k", sorted(SETS))
def test_parameter_files_are_the_published_m6(k):
    d = json.loads((paths.DATA / SETS[k]).read_text())
    assert d["model"] == "m6" and d["actuator"] == f"mangdang_md300_id_{k}"
    up = UPSTREAM / "bam" / "params" / f"mangdang_md300_id_{k}" / "m6.json" if UPSTREAM else None
    if up is not None and up.is_file():
        assert up.read_bytes() == (paths.DATA / SETS[k]).read_bytes()


def ours(k, vin=12.0):
    m = load_model(str(paths.DATA / SETS[k]))
    act = m.actuator
    assert type(act).__name__ == "MD300Actuator"
    choice.apply_md300_firmware(act, {**choice.MD300_FIRMWARE, "vin": vin})
    return m


@pytest.fixture(scope="module")
def upstream():
    """BAM's ``bam.mangdang.actuator`` from the upstream checkout (``bam.model`` is not imported: its registry pulls in
    the other servo families' dependencies)."""
    if UPSTREAM is None or not (UPSTREAM / "bam" / "mangdang" / "actuator.py").is_file():
        pytest.skip("BAM_UPSTREAM: no upstream bam checkout named")
    sys.path.insert(0, str(UPSTREAM))
    try:
        yield importlib.import_module("bam.mangdang.actuator"), importlib.import_module("bam.testbench")
    finally:
        sys.path.remove(str(UPSTREAM))


def theirs(mods, k, vin=12.0):
    """BAM's own MD300Actuator with the published m6 set, configured from a log as ``bam.fit`` does (``load_log``), at
    the robot's runner gains. Returns (actuator, its parameters)."""
    actuator_mod, testbench_mod = mods
    data = json.loads((UPSTREAM / "bam" / "params" / f"mangdang_md300_id_{k}" / "m6.json").read_text())
    act = actuator_mod.MD300Actuator(testbench_mod.Pendulum)
    act.set_model(SimpleNamespace())
    for name, par in vars(act.model).items():
        par.value = data[name]
    log = {"mass": 0.1, "arm_mass": 0.02, "length": 0.1, "kp": choice.ROBOT_KP, "at32_pwm_scale": choice.ROBOT_KFF,
           "at32_max_pwm_duty_cycle": choice.ROBOT_MAX_PWM, "cur_cap_ma": choice.ROBOT_CAP_MA,
           "at32_kd_position": choice.ROBOT_KD, "supply_v": vin}
    act.load_log(log)
    return act


@pytest.mark.parametrize("vin", [12.0, 10.5])
@pytest.mark.parametrize("k", sorted(SETS))
def test_torque_and_current_match_bam(upstream, k, vin):
    a, b = ours(k, vin), theirs(upstream, k, vin)
    assert type(b).__module__ == "bam.mangdang.actuator"
    for name in ("kt", "kt_emf", "R", "error_gain_ratio", "kd_ratio"):
        assert getattr(a, name).value == getattr(b.model, name).value, name
    for name in ("kp", "kd", "kd_scale", "error_gain", "max_pwm", "vin", "deadband"):
        assert float(getattr(a.actuator, name)) == pytest.approx(float(getattr(b, name)), rel=1e-12), name
    q0 = 0.3
    for err, dq in POINTS:
        ca = a.actuator.compute_control(q0 + err, q0, dq, 0.001)
        cb = b.compute_control(q0 + err, q0, dq, 0.001)
        ta = a.actuator.compute_torque(ca, True, q0, dq)
        tb = b.compute_torque(cb, True, q0, dq)
        assert ca == pytest.approx(cb, rel=1e-12, abs=1e-15) and ta == pytest.approx(tb, rel=1e-12, abs=1e-15)
        # the plant law by hand: I = (V - kt_emf dq) / R, tau = kt I, no knee
        p = {n: getattr(a, n).value for n in ("kt", "kt_emf", "R")}
        cur = (ca - p["kt_emf"] * dq) / p["R"]
        assert ta == pytest.approx(p["kt"] * cur, rel=1e-12, abs=1e-15)
        if abs(err) <= math.radians(0.1) and dq == 0.0:
            assert abs(ca) < 1e-9                                # the dead band (0.1 deg exactly: rounding)
        duty = ca / vin
        assert abs(duty) <= choice.ROBOT_MAX_PWM + 1e-12


def test_firmware_law_by_hand():
    """duty = pwm_scale kp (err_deg - 0.1) - kd 2.87e-5 (pwm_scale/0.001) kd_ratio dq, x error_gain_ratio on the P."""
    m = ours(2)
    act = m.actuator
    egr, kdr = m.error_gain_ratio.value, m.kd_ratio.value
    err_deg, dq = 3.0, 0.7
    duty = 0.001 * 80.0 * (err_deg - 0.1) * egr - 400.0 * 2.87e-5 * kdr * dq
    ctl = act.compute_control(math.radians(err_deg), 0.0, dq, 0.001)
    assert ctl == pytest.approx(12.0 * duty, rel=1e-9)
    assert act.compute_control(math.radians(90.0), 0.0, 0.0, 0.001) == pytest.approx(12.0 * 0.99)


def test_options_and_flags():
    ap = ArgumentParser()
    choice.add_args(ap, evaluation=True)
    a = ap.parse_args([])
    assert a.actuator is None and choice.options_from_args(a) == {
        "supply_v_range": None, "supply_drop_ohm_range": None, "r_scale_range": None}
    a = ap.parse_args(["--actuator", "md300", "--supply-v", "10.5", "12.6", "--r-scale", "1.0", "1.4"])
    assert choice.options_from_args(a) == {"actuator": "md300", "supply_v_range": (10.5, 12.6),
                                           "supply_drop_ohm_range": None, "r_scale_range": (1.0, 1.4)}
    with pytest.raises(ValueError):
        choice.check_name("md02")
    with pytest.raises(ValueError):
        choice._range((12.6, 10.5))
    from mp4_recovery import limits as L
    assert L.eval_plant({})["actuator"] == "md300"
    assert L.eval_plant({"actuator": "md300"})["actuator"] == "md300"
    assert L.eval_plant({"actuator": "md300"}, actuator={"actuator": "md01"})["actuator"] == "md01"
    assert L.eval_plant({}, actuator=choice.options_from_args(a))["supply_v_range"] == (10.5, 12.6)


def test_policy_meta_names_the_actuator(tmp_path):
    """``with_meta`` (export_policy --from-policy) carries ``actuator``; the shipped walker names md300."""
    from mp4_recovery.policy_file import NumpyPolicy, with_meta
    src = paths.POLICIES / "walker"
    if not (src / "policy.npz").is_file():
        pytest.skip("no published walker")
    assert NumpyPolicy(src / "policy.npz").meta["actuator"] == "md300"
    p = with_meta(src, tmp_path / "w", actuator="md01")
    assert NumpyPolicy(p).meta["actuator"] == "md01"


def test_dead_band_and_saturation_numpy_arrays():
    """The class runs on arrays as in a batched fit (numpy backend)."""
    act = ours(3).actuator
    err = np.radians(np.array([0.0, 0.05, 0.2, 100.0]))
    ctl = act.compute_control(err, np.zeros(4), np.zeros(4), 0.001)
    assert ctl[0] == 0.0 and ctl[1] == 0.0 and ctl[2] > 0.0 and ctl[3] == pytest.approx(12.0 * 0.99)
