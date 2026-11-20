"""The legs on the GPU: the walking task's and the get-up task's envs build and run with both feet, and the robot holding
the stand settles below the kinematic stand height by the servos' sag and the feet's contact."""
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mp4_sim.robot import model as sku2

from mp4_rl.common.env import TerrainOptions, foot_r, make_terrain_env_cfg, noise_profile
from mp4_rl.getup.task import GetupOptions, make_getup_env_cfg

FEET = ("sphere", "mesh")


def _terrain_env(n, **kw):
    opt = TerrainOptions(num_envs=n, noisy=False, noise_ramp=False, randomise=False, terrain="flat", rows=2, tile=4.0,
                         level=0, **kw)
    cfg = make_terrain_env_cfg(opt)
    cfg.seed = 1
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0"), opt


def _settled_z(n=8, seconds=2.0, **kw):
    env, opt = _terrain_env(n, **kw)
    env.reset()
    z0 = float((env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean())
    a = torch.zeros(n, 12, device="cuda:0")
    for _ in range(int(round(seconds / env.step_dt))):
        env.step(a)
    p = env.scene["robot"].data.root_link_pos_w
    assert torch.isfinite(p).all() and torch.isfinite(env.scene["robot"].data.joint_pos).all()
    assert int(env.termination_manager.terminated.sum()) == 0
    z = float((p[:, 2] - env.scene.env_origins[:, 2]).mean())
    env.close()
    return z0, z, opt


def test_the_robot_settles_at_the_kinematic_stand():
    for foot in FEET:
        z0, z, opt = _settled_z(foot=foot)
        legs = sku2.legs_of(opt)
        assert z0 == pytest.approx(sku2.init_base_z(legs), abs=1e-6)
        # sag: MD-300 statics under 630.5 g 1.5-2.3 mm (P gain past a 0.1 deg dead band, +-load friction) + feet 0.2 mm
        assert 0.0015 < sku2.stand_geometry(legs)["base_z_m"] - z < 0.0025, (foot, z)
        assert foot_r(opt) == pytest.approx(0.00411, abs=1e-5)


@pytest.mark.parametrize("foot", FEET)
def test_getup_env_with_each_foot(foot):
    n = 16
    opt = GetupOptions(num_envs=n, noisy=True, noise_ramp=False, randomise=True, noise_end=noise_profile(), foot=foot)
    cfg = make_getup_env_cfg(opt)
    cfg.seed = 3
    stand_z = sku2.init_base_z(sku2.Legs(foot=foot))
    assert cfg.rewards["height_progress"].params["stand_z"] == pytest.approx(stand_z)
    assert cfg.events["getup_start"].params["stand_z"] == pytest.approx(stand_z)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    m = env.sim.mj_model
    assert int(m.geom("robot/FR_foot").type[0]) == (2 if foot == "sphere" else 7)  # mjGEOM_SPHERE, mjGEOM_MESH
    g = torch.Generator(device="cuda:0").manual_seed(0)
    for _ in range(50):
        _, rew, *_ = env.step(torch.rand(n, 12, device="cuda:0", generator=g) * 2 - 1)
        assert torch.isfinite(rew).all()
    assert torch.isfinite(env.scene["robot"].data.root_link_pos_w).all()
    env.close()
