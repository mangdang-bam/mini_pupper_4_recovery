"""Get-ups on terrain: the ground grid against MuJoCo's rays, spawns, the up rule, and how the evaluator counts resets."""
from pathlib import Path

import mujoco
import numpy as np
import torch
from mp4_sim.envs import terrain as tm
from mp4_sim.envs import terrain_spawn as ts

from mp4_rl.skills.getup.eval import getup_eval as ge
from mp4_rl.skills.getup.task import (
    TERRAIN_FAMILIES,
    TERRAIN_ROWS,
    GetupOptions,
    base_z,
    make_getup_env_cfg,
    terrain_options,
    terrain_up_now,
)

REPO = Path(__file__).resolve().parents[2]
ALL = ("flat",) + TERRAIN_FAMILIES


def _env(n=64, **kw):
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_rl.common.env import noise_profile
    terrain = kw.pop("terrain", None) or {"terrain": "families", "families": ALL, "rows": 4, "difficulty_max": 0.4}
    cfg = make_getup_env_cfg(GetupOptions(num_envs=n, noise_end=noise_profile(), **terrain, **kw))
    cfg.seed = 3
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def test_family_columns_and_the_training_mix():
    o = terrain_options()
    cols = tm.family_columns(o["families"])
    fams = [tm.family_of(c) for c in cols]
    assert fams.count("flat") == len(cols) / 2 == len(TERRAIN_FAMILIES)
    assert set(fams) == set(ALL)
    assert o["rows"] == TERRAIN_ROWS == 8 and o["difficulty_max"] * tm.LEVELS == TERRAIN_ROWS
    four = cols["stairs4"]
    assert [tuple(r) for r in four.layout] == [(0.025, 0.05)] * 4 and four.width == 0.4
    assert len(cols["stairs8"].layout) == 8
    assert tm.family_columns(("low_poly",), relief_pitch=0.02)["low_poly"].pitch == 0.02


def test_ground_grid_matches_mujoco_rays_at_its_nodes():
    env = _env(4)
    m, ground = env.sim.mj_model, env.terrain_spawn.ground
    d = mujoco.MjData(m)
    d.qpos[2] = 50.0
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    nx, ny = ground.heights.shape
    g = np.array([-1], dtype=np.int32)
    h = ground.heights.cpu().numpy()
    origin = ground.origin.cpu().numpy()

    def ray(x, y):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0, 0, -1.0]), None, 1, -1, g)
        return 3.0 - dist if dist >= 0 else 0.0

    near = [(0.0, 0.0), (2e-3, 0.0), (-2e-3, 0.0), (0.0, 2e-3), (0.0, -2e-3)]     # a node can lie on a box's edge
    err = []
    for i, j in zip(rng.integers(1, nx - 1, 4000), rng.integers(1, ny - 1, 4000)):
        x, y = origin + (i, j) * np.array(ground.pitch)
        err.append(min(abs(ray(x + dx, y + dy) - h[i, j]) for dx, dy in near))
    err = np.array(err)
    # a heightfield's cell is two triangles, the grid bilinear: they part on a steep cell, as at an obstacle's edge
    assert (err < 1e-3).mean() > 0.99 and err.max() < 0.02, (float((err < 1e-3).mean()), float(err.max()))
    env.close()


def test_spawns_cover_the_rows_off_the_platforms_and_above_the_ground():
    env = _env(512)
    env.reset()
    spawn = env.terrain_spawn
    origins = env.scene.terrain.terrain_origins[spawn.row, spawn.col]
    d = (env.scene.env_origins - origins)[:, :2]
    assert set(spawn.row.tolist()) == set(range(4)) and set(spawn.col.tolist()) == set(range(len(ALL)))
    assert d[spawn.flat].abs().max() == 0.0
    printed = torch.tensor([spawn.families[c] in tm.PRINTED_STAIRS for c in spawn.col.tolist()], device=env.device)
    rough_tile = ~spawn.flat & ~printed
    assert d[rough_tile].abs().amax(-1).min() > ts.PLATFORM_CLEAR and d[rough_tile].abs().max() <= 0.55
    assert d[printed].abs().max() < 0.6
    p = env.scene["robot"].data.root_link_pos_w
    assert (base_z(env) > 0.0).all()
    assert torch.equal(base_z(env), p[:, 2] - spawn.ground.under(p[:, :2]))
    env.close()


def test_local_height_on_flat_is_the_world_height_and_the_critic_sees_it():
    env = _env(64)
    obs, _ = env.reset()
    spawn = env.terrain_spawn
    p = env.scene["robot"].data.root_link_pos_w
    assert (spawn.ground.under(p[spawn.flat][:, :2]) == 0.0).all()
    assert torch.equal(base_z(env)[spawn.flat], p[spawn.flat, 2])
    om = env.observation_manager
    term = om._group_obs_term_cfgs["critic"][om._group_obs_term_names["critic"].index("height")]
    assert torch.equal(term.func(env)[:, 0], base_z(env))
    env.close()


def test_up_rule_tilt_and_feet():
    env = _env(8, terrain={"terrain": "families", "families": ("low_poly", "stairs4"), "rows": 1, "difficulty_max": 0.1})
    env.reset()
    robot, spawn = env.scene["robot"], env.terrain_spawn
    n = env.num_envs
    spawn.fix_cells(torch.zeros(n, dtype=torch.long, device=env.device), torch.arange(n, device=env.device) % 2)
    env.reset()
    assert spawn.on_stairs.tolist() == [False, True] * 4
    # a tilt of 30 deg on stairs and off them, at the stand, on the ground: only the feet and the tilt differ
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    zero = torch.zeros(n, device=env.device)
    root = torch.cat([robot.data.root_link_pos_w,
                      quat_from_euler_xyz(zero + np.radians(30.0), zero, zero), torch.zeros(n, 6, device=env.device)], -1)
    robot.write_root_state_to_sim(root)
    from mp4_rl.common import mdp
    act = mdp.servo_action(env)
    q = act.joint_state_from_rel(torch.zeros(n, 12, device=env.device), torch.arange(n, device=env.device))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=act._jid_all)
    env.sim.forward()
    env.scene.update(0.0)
    found = env.scene["feet_contact"].data.found
    found[:] = 1
    up = terrain_up_now(env, spawn.on_stairs)
    assert up.tolist() == [False, True] * 4
    found[:] = 0
    assert not terrain_up_now(env, spawn.on_stairs).any()
    env.close()


def test_up_tracker_counts_a_reset_as_a_failure():
    dev = "cpu"
    up = ge.UpTracker(4, hold=3, device=dev)
    ok = torch.ones(4, dtype=torch.bool)
    resets = {1: [0], 5: [1]}          # robot 0 resets before it is up, robot 1 after it, 2 never, 3 is never ok
    ok[3] = False
    for k in range(10):
        reset = torch.zeros(4, dtype=torch.bool)
        for i in resets.get(k, []):
            reset[i] = True
        up.update(ok, reset, (k + 1) * 0.02)
    assert up.t_up.tolist()[0] == -1.0 and up.t_up.tolist()[1] == -1.0
    assert abs(up.t_up.tolist()[2] - 0.06) < 1e-6 and up.t_up.tolist()[3] == -1.0
    assert up.out.tolist() == [True, True, False, False]


def test_terrain_summary_and_cells():
    cols = ["low_poly", "low_poly.1", "stairs4"]
    cells = ge.terrain_cells(cols, [3, 6])
    fam = [tm.family_of(cols[c]) for _, c in cells]
    assert fam.count("low_poly") == fam.count("stairs4") == 4
    tr = {"t_up": np.array([1.0, -1.0, -1.0, -1.0]), "t_strict": np.array([1.0, -1.0, -1.0, -1.0]),
          "reset": np.array([False, True, False, False]), "final_tilt_deg": np.array([5.0, 5.0, 170.0, 90.0]),
          "restarts": np.zeros(4), "peak_A": np.zeros(4), "drop_m": np.zeros(4), "moved_m": np.zeros(4),
          "height_m": np.full(4, 0.07)}
    s = ge.terrain_summary(tr, np.ones(4, dtype=bool), 5.0)
    assert (s["up"], s["never_up"], s["resets"], s["down_on_back"], s["down_on_side"], s["down_low_tilt"]) == (0.25, 3, 1, 1, 1, 0)


def test_evaluator_fails_a_robot_that_blows_up_and_does_not_retry_it():
    pdir = REPO / "mp4_rl" / "skills" / "getup" / "policies" / "sku2"
    env, opt = ge.build(pdir, 16, 12, terrain=ge.terrain_kwargs(["low_poly"], [1]))
    pol = ge.TorchPolicy(pdir / "policy.npz", "cuda:0")
    robot = env.scene["robot"]
    spawn = env.terrain_spawn
    spawn.fix_cells(torch.ones(16, dtype=torch.long, device=env.device), torch.arange(16, device=env.device) % len(spawn.columns))

    def fling(k):
        if k == 10:
            v = robot.data.root_link_vel_w.clone()
            v[:4, 2] = 50.0
            robot.write_root_link_velocity_to_sim(v)

    tr = ge.run_terrain_start(env, pol, "crouch", 8.0, 12, 5.0, step_hook=fling)
    assert tr["reset"][:4].all() and not tr["reset"][4:].any()
    assert (tr["t_up"][:4] == -1.0).all() and (tr["restarts"][:4] == 0).all()
    s = ge.terrain_summary(tr, np.ones(16, dtype=bool), 5.0)
    assert s["resets"] == 4 and s["never_up"] >= 4
    assert (tr["t_up"][4:] >= 0).sum() >= 8          # the others rise from a crouch
    env.close()


def test_flat_build_has_no_terrain_spawn():
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_rl.common.env import noise_profile
    cfg = make_getup_env_cfg(GetupOptions(num_envs=4, noise_end=noise_profile()))
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    assert not hasattr(env, "terrain_spawn")
    assert cfg.sim.nconmax == GetupOptions.nconmax and cfg.sim.njmax == GetupOptions.njmax
    assert torch.equal(base_z(env), env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])
    env.close()
