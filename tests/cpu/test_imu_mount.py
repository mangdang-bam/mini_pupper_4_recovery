"""The IMU mounting file: a named mounting or a custom rotation, and `runner imu` printing gravity under it."""
import json

import numpy as np
import pytest

from mp4_rl.getup.runner import runner as rn

CUSTOM = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]          # a quarter turn about z


def mount_file(tmp_path, **spec):
    f = tmp_path / "imu_mount.json"
    f.write_text(json.dumps(spec))
    return f


def test_a_named_mount_as_before(tmp_path):
    name, R = rn.load_mount(path=mount_file(tmp_path, mount="flip_y", R_sensor_to_body=CUSTOM))
    assert name == "flip_y" and np.array_equal(R, rn.IMU_MOUNTS["flip_y"])
    assert rn.load_mount()[0] in rn.IMU_MOUNTS                                  # the shipped file names one
    with pytest.raises(ValueError, match="unknown IMU mount"):
        rn.load_mount(path=mount_file(tmp_path, mount="upside"))


def test_a_custom_mount_reads_its_matrix(tmp_path):
    name, R = rn.load_mount(path=mount_file(tmp_path, mount="custom", R_sensor_to_body=CUSTOM))
    assert name == "custom" and np.array_equal(R, np.array(CUSTOM, dtype=float))


@pytest.mark.parametrize("m, why", [
    (None, "3 x 3"),
    ([[1, 0], [0, 1]], "3 x 3"),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]], "-1, 0 or 1"),
    ([[1, 0, 0], [1, 0, 0], [0, 0, 1]], "proper rotation"),             # not orthonormal
    ([[1, 0, 0], [0, 1, 0], [0, 0, -1]], "proper rotation"),            # a reflection, det -1
])
def test_a_custom_mount_must_be_a_proper_rotation(tmp_path, m, why):
    with pytest.raises(ValueError, match=why):
        rn.load_mount(path=mount_file(tmp_path, mount="custom", R_sensor_to_body=m))


def test_imu_prints_gravity_under_the_custom_mount(robot, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rn, "IMU_MOUNT_JSON", mount_file(tmp_path, mount="custom", R_sensor_to_body=CUSTOM))
    robot.acc = [0, 1000, 0]                                     # sensor y up: custom R -> body x up, gravity -x
    assert rn.main(["--port", robot.port, "imu", "--seconds", "0.3", "--log", str(tmp_path / "x" / "imu.npz")]) == 0
    out = capsys.readouterr().out
    assert "custom: g=(+1.00, " in out
    assert "swap_xy:" in out and (tmp_path / "x" / "imu.npz").exists()
    monkeypatch.setattr(rn, "IMU_MOUNT_JSON", mount_file(tmp_path, mount="swap_xy"))
    assert rn.main(["--port", robot.port, "imu", "--seconds", "0.2"]) == 0
    assert "custom:" not in capsys.readouterr().out
