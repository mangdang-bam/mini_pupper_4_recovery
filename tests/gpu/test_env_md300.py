"""The md300 servo model in the environment: its gains, its torques against BAM's class on the CPU, a held stand, and the
supply / winding randomisation."""
import math

import mujoco
import numpy as np
import pytest
import torch
from mp4_sim.actuators import choice, servo_model
from mp4_sim.actuators.bam.model import load_model

from mp4_rl.common.env import TaskOptions, describe, make_env_cfg

N = 16


def make(**kw):
    from mjlab.envs import ManagerBasedRlEnv
    cfg = make_env_cfg(TaskOptions(**kw))
    cfg.seed = 3
    e = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    e.reset()
    return e


@pytest.fixture(scope="module")
def env():
    e = make(num_envs=N, actuator="md300")
    yield e
    e.close()


def cpu_model(path, vin=12.0, r_scale=1.0):
    m = load_model(str(path))
    choice.apply_md300_firmware(m.actuator, {**choice.MD300_FIRMWARE, "vin": vin})
    m.R.value *= r_scale
    return m


def cpu_torque(m, target, pos, vel):
    act = m.actuator
    return act.compute_torque(act.compute_control(target, pos, vel, 0.001), True, pos, vel)


def test_md300_actuators_echo(env):
    m = env.sim.mj_model
    acts = env.scene["robot"].actuators
    assert [type(a).__name__ for a in acts] == ["BamServo"] * 3
    assert servo_model.md300().latency_steps == (5, 4, 3)          # 4.04, 3.02, 2.73 ms rounded up
    J = {role: float(load_model(str(path)).actuator.get_extra_inertia()) for role, _, path, _ in servo_model.MD300_ROLES}
    for leg in ("FR", "FL", "RR", "RL"):
        for joint, role in (("hip", "abd"), ("thigh", "hip")):
            dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_{joint}")]
            assert m.dof_armature[dof] == pytest.approx(J[role])
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f"robot/{leg}_knee_drive")
        assert m.tendon_armature[t] == pytest.approx(J["knee"])
    for a, (_, _, path, _) in zip(acts, servo_model.MD300_ROLES):
        act = a._bam_model.actuator
        assert type(act).__name__ == "MD300Actuator" and a.cfg.json_path == str(path)
        assert (a._base_kp, act.kd, act.kd_scale, act.max_pwm) == (80.0, 400.0, 1.0, 0.99)
        assert act.error_gain == pytest.approx(0.001 * 180.0 / math.pi, rel=1e-12)
        assert act.deadband == pytest.approx(math.radians(0.1))
        assert torch.all(a.vin_tensor == 12.0) and a.supply.drop is None and a.supply.r_scale is None
    assert acts[0].supply is acts[1].supply is acts[2].supply
    d = describe(TaskOptions(actuator="md300"))
    assert d["servo_model"] == "md300" and d["options"]["actuator"] == "md300"
    assert d["servo_params"]["firmware"]["pwm_scale"] == 0.001
    assert d["servo_params"]["roles"]["knee"]["params"].endswith("md300_id_3_m6.json")


def test_md300_holds_the_stand_and_matches_bam(env):
    """200 policy steps (4 s) of the stand: finite, nothing ends, upright; each group's torque is BAM's class on the CPU
    at the same target, position and velocity."""
    zero = torch.zeros(N, 12, device="cuda:0")
    ended = torch.zeros(N, dtype=torch.bool, device="cuda:0")
    robot = env.scene["robot"]
    for _ in range(200):
        _, _, term, trunc, _ = env.step(zero)
        ended |= term | trunc
        assert torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all()
    assert not ended.any()
    g = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
    assert float(tilt.max()) < 10.0, tilt
    h = robot.data.root_link_pos_w[:, 2]                          # the stand's base is at 71.9 mm
    assert float(h.min()) > 0.055, h
    for a, (_, _, path, _) in zip(env.scene["robot"].actuators, servo_model.MD300_ROLES):
        last = a._last
        m = cpu_model(path)
        for i in range(0, N, 5):
            t, p, v = (last[k][i].double().cpu().numpy() for k in ("target", "pos", "vel"))
            want = cpu_torque(m, t, p, v)
            got = last["torque"][i].double().cpu().numpy()
            assert np.allclose(got, want, rtol=1e-4, atol=1e-5), (got, want)


def test_supply_and_winding_randomisation():
    """One supply per robot, shared by its three groups; drawn in range; the sag lowers the bus with the draw; the
    winding scale is applied inside the step and the fitted R restored after it."""
    n = 64
    e = make(num_envs=n, actuator="md300", supply_v_range=(10.5, 12.6), supply_drop_ohm_range=(0.2, 0.4),
             r_scale_range=(1.0, 1.4))
    try:
        acts = e.scene["robot"].actuators
        sup = acts[0].supply
        assert all(a.supply is sup for a in acts)
        assert float(sup.vin.min()) >= 10.5 and float(sup.vin.max()) <= 12.6 and float(sup.vin.std()) > 0.1
        assert float(sup.r_scale.min()) >= 1.0 and float(sup.r_scale.max()) <= 1.4
        assert float(sup.drop.min()) >= 0.2 and float(sup.drop.max()) <= 0.4
        g = torch.Generator(device="cuda:0").manual_seed(0)
        for _ in range(15):
            e.step(0.5 * torch.randn(n, 12, device="cuda:0", generator=g))
        for a, (_, _, path, _) in zip(acts, servo_model.MD300_ROLES):
            assert a._bam_model.R.value == a._R0                         # restored after the step
            vin = a._last["vin"]
            assert torch.all(vin <= sup.vin + 1e-6)
            for i in range(0, n, 9):
                m = cpu_model(path, vin=float(vin[i]), r_scale=float(sup.r_scale[i]))
                t, p, v = (a._last[k][i].double().cpu().numpy() for k in ("target", "pos", "vel"))
                assert np.allclose(a._last["torque"][i].double().cpu().numpy(), cpu_torque(m, t, p, v),
                                   rtol=1e-4, atol=1e-5)
        assert float((sup.vin - acts[0]._last["vin"]).max()) > 0.01      # drawing current sags the bus
        assert float(sup.bus_current.max()) > 0.0
    finally:
        e.close()
