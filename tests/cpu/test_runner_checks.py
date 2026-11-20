"""The refusals: policies the robot must not run and targets it must not send; the shipped policies pass."""
import json

import numpy as np
import pytest
from mp4_sim import robots as sim_robots
from robot_helpers import PACKAGED, C, getup_policy, go, make_policy

from mp4_rl import paths
from mp4_rl.common.policy import (
    CONTRACT_JSON,
    ActionDecoder,
    Bounds,
    contract_tendon_coefs,
    policy_bounds,
)
from mp4_rl.getup.runner import checks
from mp4_rl.getup.runner import runner as rn

WALKER, GETUP = paths.WALKER_POLICIES / "sku2", paths.GETUP_POLICIES / "sku2"


def real_meta(path):
    return json.loads(str(np.load(path / "policy.npz")["meta"]))


@pytest.mark.parametrize("policy", [WALKER, GETUP])
def test_shipped_policies_pass_check(policy, capsys):
    assert rn.main(["check", "--policy", str(policy)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "calibration: ok (sku2_contract.json" in out


def test_shipped_policies_observe_the_standard_layout():
    w, g = rn.NumpyPolicy(WALKER / "policy.npz"), rn.NumpyPolicy(GETUP / "policy.npz")
    assert rn.layout_refusal(w) is None and rn.layout_refusal(g) is None
    assert rn.policy_kind(w.meta) == "walk" and rn.policy_kind(g.meta) == "getup" and g.meta["getup_phase_s"] > 0


def test_check_fails_on_wrong_fixtures(fixed_calibration, tmp_path, capsys):
    d = make_policy(tmp_path / "p")
    z = dict(np.load(d / "policy.npz"))
    z["fixture_act"] = z["fixture_act"] + 1.0
    np.savez(d / "policy.npz", **z)
    assert rn.main(["check", "--policy", str(d)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_fails_on_another_contract(fixed_calibration, tmp_path, capsys):
    d = make_policy(tmp_path / "p", contract_sha256="0" * 64)
    assert rn.main(["check", "--policy", str(d)]) == 1
    assert "calibration: FAIL: the policy was exported for the calibration 000000000000" in capsys.readouterr().out


# ---- the calibration each policy folder carries -------------------------------------------------------------------
def test_check_and_run_refuse_a_folder_without_its_calibration(robot, tmp_path, capsys):
    w = make_policy(tmp_path / "w", calibration=None)
    g = getup_policy(tmp_path / "g")
    assert rn.main(["check", "--policy", str(w)]) == 1
    assert "calibration: FAIL" in capsys.readouterr().out
    rc = rn.main(["--port", robot.port, "run", "--policy", str(w), "--getup-policy", str(g), "--stand-ramp", "0.2"])
    out = capsys.readouterr().out
    assert rc == 2 and "refused: --policy:" in out and "no calibration file" in out and not robot.armed_seen


def test_run_refuses_an_altered_calibration(robot, tmp_path, capsys):
    w = make_policy(tmp_path / "w")
    g = getup_policy(tmp_path / "g")
    f = g / "sku2_contract.json"
    d = json.loads(f.read_text())
    d["offsets_deg"][0] += 1.0                  # an edited calibration: neither the exported one nor one it replaced
    f.write_text(json.dumps(d))
    rc = rn.main(["--port", robot.port, "run", "--policy", str(w), "--getup-policy", str(g), "--stand-ramp", "0.2"])
    out = capsys.readouterr().out
    assert rc == 2 and "refused: --getup-policy: the policy was exported for the calibration" in out
    assert not robot.armed_seen
    assert rn.main(["check", "--policy", str(g)]) == 1


def test_run_refuses_a_walker_and_a_get_up_with_different_calibrations(robot, tmp_path, capsys):
    w = make_policy(tmp_path / "w")
    g = getup_policy(tmp_path / "g", calibration=CONTRACT_JSON)
    rc = rn.main(["--port", robot.port, "run", "--policy", str(w), "--getup-policy", str(g), "--stand-ramp", "0.2"])
    out = capsys.readouterr().out
    assert rc == 2 and "refused: the walking and the get-up policy carry different calibrations" in out
    assert not robot.armed_seen


def test_a_calibration_that_replaced_the_exported_one_is_accepted(fixed_calibration, tmp_path, capsys):
    """A folder whose calibration lists the one the policy was exported for (``earlier_calibrations``) passes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_contract", paths.REPO / "tools" / "make_contract.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    from robot_helpers import TEST_CONTRACT_JSON
    w = make_policy(tmp_path / "w")
    new = tool.build(TEST_CONTRACT_JSON, C.offsets_deg + 1.0, "recalibrated")
    (w / "sku2_contract.json").write_text(json.dumps(new, indent=1) + "\n")
    assert rn.main(["check", "--policy", str(w)]) == 0
    assert "calibration: ok (exported for an earlier calibration)" in capsys.readouterr().out


def test_a_run_logs_the_calibration_and_the_imu_mount(robot, tmp_path):
    from robot_helpers import TEST_CONTRACT_JSON
    rc, _, meta = go(robot, tmp_path, seconds=1.0)
    assert rc == 0
    assert meta["contract_sha256"] == paths.sha256(TEST_CONTRACT_JSON) and meta["calibration_file"] == "sku2_contract.json"
    assert meta["mount_file_sha256"] == paths.sha256(rn.IMU_MOUNT_JSON)


def test_bands_the_robot_refuses():
    assert "wider than the robot's 90" in checks.band_refusal({"band_deg": [91.0] * 12})
    assert "walking band" in checks.band_refusal({"band_deg": [30.0] * 12})
    assert "only with decoder bounds" in checks.band_refusal({"kind": "getup", "band_deg": [120.0] * 12})
    assert "unknown policy kind" in checks.band_refusal({"kind": "dance"})
    assert checks.band_refusal({"kind": "getup", "band_deg": [90.0] * 12}) is None
    assert checks.band_refusal({}) is None


def test_bounds_must_match_the_contract_calibration():
    meta = real_meta(GETUP)
    assert checks.band_refusal(meta) is None
    meta["bounds"]["info"]["contract_stand_dd"] = [v + 10 for v in PACKAGED.stand_dd]
    assert "another calibration" in checks.band_refusal(meta)


def test_a_stand_offset_needs_bounds():
    assert "without decoder bounds" in checks.band_refusal({"stand_offset_abd_deg": 12.0, "band_deg": [20, 60, 60] * 4})


def test_run_refuses_policies_of_the_wrong_kind(robot, tmp_path, capsys):
    g = getup_policy(tmp_path / "g")
    rc = rn.main(["--port", robot.port, "run", "--policy", str(g), "--getup-policy", str(g)])
    assert rc == 2 and "not a walking policy" in capsys.readouterr().out
    w = make_policy(tmp_path / "w")
    rc = rn.main(["--port", robot.port, "run", "--policy", str(w), "--getup-policy", str(w)])
    assert rc == 2 and "not a get-up policy" in capsys.readouterr().out
    assert not robot.armed_seen


def test_run_refuses_a_get_up_without_its_phase_clock(robot, tmp_path, capsys):
    rc = go(robot, tmp_path, getup_meta={"getup_phase_s": 0.0})[0]
    assert rc == 2 and "phase clock" in capsys.readouterr().out and not robot.armed_seen


def test_run_refuses_a_walker_without_the_whole_command(robot, tmp_path, capsys):
    rc = go(robot, tmp_path, walker_meta={"omni": False})[0]
    assert rc == 2 and "whole command" in capsys.readouterr().out


def test_target_refusal_near_a_track_end():
    dec = ActionDecoder(C, np.full(12, 0.3))
    assert checks.target_refusal(dec, {"rel_exact": np.zeros(12), "dd": np.asarray(C.stand_dd)}) is None
    dd = np.array(C.stand_dd)
    dd[4] = 3000
    assert "servo 5" in checks.target_refusal(dec, {"rel_exact": np.zeros(12), "dd": dd})


def test_target_refusal_outside_the_bounds():
    b = policy_bounds(real_meta(GETUP))
    dec = ActionDecoder(PACKAGED, np.full(12, 1.57), band_deg=[90.0] * 12, bounds=b)
    far = np.full(12, 4.0)
    assert "outside the policy's bounds" in checks.target_refusal(dec, {"rel_exact": far, "dd": None})


def test_selftest_passes_the_shipped_policies_and_catches_a_dead_zone():
    for path in (WALKER, GETUP):
        pol = rn.NumpyPolicy(path / "policy.npz")
        assert checks.decoder_selftest(rn.policy_decoder(pol, PACKAGED), PACKAGED) is None
    wide = Bounds(lo=C.stand_rel_rad - 4.0, hi=C.stand_rel_rad + 4.0, kc=contract_tendon_coefs(C),
                  calf_lo=np.full(4, -10.0), calf_hi=np.full(4, 10.0))
    dec = ActionDecoder(C, np.full(12, 3.0), band_deg=[170.0] * 12, bounds=wide)
    assert "track end" in checks.decoder_selftest(dec, C)


def test_selftest_failure_refuses_before_arming(robot, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rn, "decoder_selftest", lambda dec, c: "scripted")
    assert go(robot, tmp_path)[0] == 2
    assert "decoder self-test: scripted" in capsys.readouterr().out and not robot.armed_seen


def test_the_contract_stand_is_its_calibration():
    for c in (PACKAGED, C):
        assert c.dd_from_ticks(c.ticks_from_servo_deg(np.zeros(12))).tolist() == list(c.stand_dd)


def test_a_band_only_policy_is_refused_at_the_packaged_calibration(packaged_robot, tmp_path, capsys):
    """At the packaged calibration a stand is within the walking band of its track end (FL thigh, servo 5):
    a policy without decoder bounds is refused, by ``check`` and by ``run`` before arming."""
    w = make_policy(tmp_path / "w", calibration=CONTRACT_JSON)
    g = getup_policy(tmp_path / "g", calibration=CONTRACT_JSON)
    assert rn.main(["check", "--policy", str(w)]) == 1
    out = capsys.readouterr().out
    assert "calibration: ok" in out and "FAIL: servo 5 target" in out and "of its track end" in out
    rc = rn.main(["--port", packaged_robot.port, "run", "--policy", str(w), "--getup-policy", str(g), "--stand-ramp", "0.2",
                  "--seconds", "1", "--log", str(tmp_path / "run")])
    out = capsys.readouterr().out
    assert rc == 2 and "refused: --policy: decoder self-test: servo 5 target" in out and "of its track end" in out
    assert not packaged_robot.armed_seen


def test_shipped_policies_run_the_sequence_on_the_fake_robot(packaged_robot, tmp_path):
    """The real networks, at the calibration they were exported for, through a fall, a get-up and a hand-back."""
    robot = packaged_robot
    from fake_esp32 import fall_and_rise, roll
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=1.0, up_s=2.0)), 100
    orig = robot.before_tick

    def tick():                                 # upright again: the joints at the stand whatever the policy asks
        orig()
        if robot.getup and robot.script_tick is not None and (robot.tick - robot.script_tick) * 0.005 > 1.3:
            robot.pos = [float(v) for v in PACKAGED.stand_dd]
    robot.before_tick = tick
    log = tmp_path / "run"
    rc = rn.main(["--port", packaged_robot.port, "run", "--seconds", "4", "--log", str(log)])
    z = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(z["meta"]))
    assert rc == 0 and meta["stop_reason"] == "completed" and meta["getups"][0]["end"] == "hand-back"
    assert meta["policy_sha256"] == paths.sha256(WALKER / "policy.npz")



# ---- the policy against the robot: servo model, legs, foot, model file -------------------------------------------------
def test_the_robot_is_the_packaged_model_and_servo_model():
    """``checks.ROBOT_*`` (what the runner compares, without the model files on the Pi) are mp4_sim's SKU2 model and
    the default servo model, and the shipped policies were trained on them."""
    from mp4_sim.actuators import choice
    from mp4_sim.robot import model as sku2
    lm = sku2.legs_meta()
    assert checks.ROBOT_ACTUATOR == choice.NAME
    assert checks.ROBOT_LEGS == {k: lm[k] for k in checks.ROBOT_LEGS}
    assert checks.ROBOT_LEGS["model_sha256"] == paths.sha256(sim_robots.SKU2.xml)
    for p in (WALKER, GETUP):
        assert checks.robot_refusal(real_meta(p)) is None


@pytest.mark.parametrize("change, why", [
    (lambda m: m.update(actuator="md01"), "servo model md01"),
    (lambda m: m.pop("actuator"), "servo model (not named)"),
    (lambda m: m.pop("legs"), "names no legs"),
    (lambda m: m["legs"].update(legs="sku_1"), "legs:"),
    (lambda m: m["legs"].update(foot="mesh"), "foot 'mesh'"),
    (lambda m: m["legs"].update(foot_radius_m=0.0049), "0.0049 m foot"),
    (lambda m: m["legs"].update(model_sha256="0" * 64), "another robot model"),
])
def test_a_policy_for_another_robot_is_refused(change, why):
    meta = real_meta(WALKER)
    change(meta)
    assert why in checks.robot_refusal(meta)


def test_the_robot_check_reads_the_legs_alias_and_ignores_paths():
    meta = real_meta(GETUP)
    meta["legs"].update(legs="new", model="/elsewhere/pupper.xml")
    assert checks.robot_refusal(meta) is None


def test_check_fails_on_a_policy_for_another_robot(fixed_calibration, tmp_path, capsys):
    d = make_policy(tmp_path / "p", actuator="md01")
    assert rn.main(["check", "--policy", str(d)]) == 1
    assert "robot: FAIL: trained on the servo model md01" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["walker", "getup"])
def test_run_refuses_a_policy_for_another_robot(robot, tmp_path, capsys, which):
    other = {"legs": {**checks.ROBOT_LEGS, "model_sha256": "0" * 64}}
    rc, _, _ = go(robot, tmp_path, **{f"{which}_meta": other})
    assert rc == 2 and not robot.armed_seen
    flag = "--policy" if which == "walker" else "--getup-policy"
    assert f"refused: {flag}: trained on another robot model" in capsys.readouterr().out


# ---- the gyro's bias at the start ------------------------------------------------------------------------------------
def test_a_still_robot_runs_with_its_small_gyro_bias(robot, tmp_path):
    robot.gyr = [-1, 5, 1]                                          # 0.1 deg/s units: at most 0.009 rad/s
    rc, _, meta = go(robot, tmp_path, seconds=0.2)
    assert rc == 0 and meta["stop_reason"] == "completed"
    assert np.abs(meta["gyro_bias_rad_s"]).max() < rn.GYRO_BIAS_MAX


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_robot_moving_during_the_bias_measurement_is_refused(robot, tmp_path, axis):
    robot.gyr = [0, 0, 0]
    robot.gyr[axis] = 15                                            # 1.5 deg/s = 0.026 rad/s
    with pytest.raises(SystemExit, match="keep the robot still"):
        go(robot, tmp_path)
    assert not robot.armed_seen
