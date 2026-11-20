"""The get-up's training options: the attachment on the hood, the rollover cost and the mid-fall rotation."""
import math

import mujoco
import numpy as np
import pytest
import torch
from mp4_sim.envs.attachment import (
    GEOM,
    HOOD_TOP_Z,
    Attachment,
    add_cylinder,
    combined_inertia,
    principal,
)
from mp4_sim.robot import model as sku2

from mp4_rl.getup.task import GetupOptions, getup_collisions, stiff_contacts


def _bare_spec():
    spec = sku2.get_spec(strip_visuals=True, self_collision=True)
    getup_collisions(spec)
    stiff_contacts(spec)
    return spec


# ------------------------------------------------------------------------------------------------------------- CPU
def test_the_hood_top_is_the_models():
    m = _bare_spec().compile()
    base = [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("base").id]
    assert abs(max(m.geom_pos[i][2] + m.geom_size[i][2] for i in base) - HOOD_TOP_Z) < 1e-9


def test_parse_a_fixed_attachment():
    att = Attachment.parse("35,18.5,0.0445,50")
    assert att == Attachment((35.0, 35.0), (50.0, 50.0), (44.5, 44.5), 18.5)
    with pytest.raises(ValueError):
        Attachment(height_mm=(40.0, 0.0))


def test_the_cylinder_is_in_the_bodys_collision_class_and_sunk_it_stays_inside():
    spec = _bare_spec()
    add_cylinder(spec, Attachment())
    m = spec.compile()
    cyl = m.geom(GEOM).id
    boxes = [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("base").id and i != cyl]
    lower, upper = sorted(boxes, key=lambda i: m.geom_pos[i][2])
    assert m.geom_bodyid[cyl] == m.geom_bodyid[upper]
    assert (m.geom_contype[cyl], m.geom_conaffinity[cyl], m.geom_priority[cyl]) == \
        (m.geom_contype[upper], m.geom_conaffinity[upper], m.geom_priority[upper])
    assert np.array_equal(m.geom_solref[cyl], m.geom_solref[upper])
    assert np.array_equal(m.geom_friction[cyl], m.geom_friction[upper])
    radius, half = m.geom_size[cyl][:2]
    assert abs(half - 0.020) < 1e-9 and abs(radius - 0.0185) < 1e-9
    assert HOOD_TOP_Z - 2 * half > m.geom_pos[lower][2] - m.geom_size[lower][2]    # its lower end stays above the belly
    assert radius < m.geom_size[upper][1]                                          # and inside the hood's width
    hood_front = m.geom_pos[upper][0] + m.geom_size[upper][0]
    assert 0.0525 + radius <= hood_front + 1e-9 < 0.055 + radius                   # the front: inside up to x = 52.5 mm


def test_no_height_adds_no_geom():
    spec = _bare_spec()
    n = spec.compile().ngeom
    add_cylinder(spec, Attachment.parse("0,18.5,0.0445,50"))
    assert spec.compile().ngeom == n


def test_combined_inertia_matches_mujocos_fused_bodies():
    spec = mujoco.MjSpec()
    spec.compiler.fusestatic = True
    a = spec.worldbody.add_body(name="a")
    a.add_freejoint()
    a.mass, a.ipos, a.fullinertia = 0.2, [0.01, -0.02, 0.03], [3e-4, 4e-4, 5e-4, 1e-5, -2e-5, 3e-5]
    a.explicitinertial = True
    c = a.add_body(name="c", pos=[0.05, 0.0, 0.03])
    c.mass, c.ipos, c.fullinertia = 0.05, [0.0, 0.0, 0.0], [1e-5, 1e-5, 8e-6, 0.0, 0.0, 0.0]
    c.explicitinertial = True
    a.add_geom(size=[0.01, 0.01, 0.01], type=mujoco.mjtGeom.mjGEOM_BOX, mass=0)
    m = spec.compile()
    assert m.nbody == 2                                   # c merged into a by the compiler
    b = m.body("a").id
    tensor0 = np.array([[3e-4, 1e-5, -2e-5], [1e-5, 4e-4, 3e-5], [-2e-5, 3e-5, 5e-4]])
    tensor1 = np.diag([1e-5, 1e-5, 8e-6])
    t = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float64)         # noqa: E731
    mass, com, tensor = combined_inertia(t(0.2), t([0.01, -0.02, 0.03]), t(tensor0), t(0.05), t([0.05, 0.0, 0.03]), t(tensor1))
    assert abs(float(mass) - m.body_mass[b]) < 1e-12
    assert np.allclose(com[0].numpy(), m.body_ipos[b], atol=1e-9)
    assert np.allclose(np.sort(torch.linalg.eigvalsh(tensor)[0].numpy()), np.sort(m.body_inertia[b]), atol=1e-9)


def test_principal_rebuilds_the_tensor():
    torch.manual_seed(0)
    from mjlab.utils.lab_api.math import matrix_from_quat
    a = torch.randn(32, 3, 3, dtype=torch.float64)
    tensor = a @ a.transpose(1, 2) + torch.eye(3, dtype=torch.float64)
    diagonal, quat = principal(tensor)
    rot = matrix_from_quat(quat)
    assert torch.allclose(rot @ torch.diag_embed(diagonal) @ rot.transpose(1, 2), tensor, atol=1e-9)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(32, dtype=torch.float64))


def test_the_options_are_off_by_default():
    o = GetupOptions()
    assert o.attachment is None and o.rollover_cost == 0.0 and tuple(o.midfall_omega) == (2.0, 6.0)
    from mp4_rl.getup.task import make_getup_env_cfg
    cfg = make_getup_env_cfg(GetupOptions(num_envs=1))
    assert "attachment" not in cfg.events and "rollover" not in cfg.rewards
    assert GEOM not in [g.name for g in cfg.scene.entities["robot"].spec_fn().geoms]


# ------------------------------------------------------------------------------------------------------------- GPU
def _env(n, seed=3, **kw):
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_rl.common.env import noise_profile
    from mp4_rl.getup.task import make_getup_env_cfg
    cfg = make_getup_env_cfg(GetupOptions(num_envs=n, noise_end=noise_profile(), **kw))
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def _ids(env, geom=False):
    robot = env.scene["robot"]
    body = robot.indexing.body_ids[robot.find_bodies("base")[0][0]]
    return (body, robot.indexing.geom_ids[robot.find_geoms(GEOM)[0][0]]) if geom else body


def test_the_cylinders_top_per_robot_matches_the_draw():
    from mp4_rl.getup.task import contact_penetration
    n = 32
    env = _env(n, attachment=Attachment(), start_joints="stand", fixed_iter=0)
    env.reset()
    body, geom = _ids(env, geom=True)
    draw = env.attachment_draw
    h, x = draw["height_m"], draw["x_m"]
    assert float(h.min()) >= 0.0 and float(h.max()) <= 0.040 and float(h.max() - h.min()) > 0.010
    assert float(x.min()) >= 0.035 and float(x.max()) <= 0.055 and float(x.max() - x.min()) > 0.005
    model = env.sim.model
    half = float(model.geom_size[0, geom, 1])
    assert abs(half - 0.020) < 1e-6
    assert torch.allclose(model.geom_pos[:, geom, 2] + half, HOOD_TOP_Z + h, atol=1e-6)
    assert torch.allclose(model.geom_pos[:, geom, 0], x, atol=1e-6)
    # lying on its back 2 mm lower than the cylinder's top: the collision itself sees each robot's height
    robot = env.scene["robot"]
    pos = env.scene.env_origins.clone()
    pos[:, 2] += HOOD_TOP_Z + h - 0.002
    quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda:0").expand(n, 4)      # roll 180 deg
    robot.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(n, 6, device="cuda:0")], -1))
    env.sim.forward()
    floor, _ = contact_penetration(env)
    assert torch.allclose(floor, torch.full((n,), 0.002, device="cuda:0"), atol=2e-4), floor
    env.close()


def test_mass_and_center_of_mass_build_on_the_plants_own_draws():
    n = 32
    bare = _env(n, seed=5)
    body = _ids(bare)
    m0, c0 = bare.sim.model.body_mass[:, body].clone(), bare.sim.model.body_ipos[:, body].clone()
    tree0 = bare.sim.model.body_subtreemass[:, body].clone()
    bare.close()
    assert float(c0.std(0).max()) > 1e-4 and float(m0.std()) > 1e-3         # the plant's draws, per robot
    env = _env(n, seed=5, attachment=Attachment())
    m = env.sim.model
    d = env.attachment_draw
    h, added, x = d["height_m"], d["mass_kg"], d["x_m"]
    assert float(added.min()) >= 0.0 and float(added.max()) <= 0.070 and float(added.max() - added.min()) > 0.02
    assert torch.allclose(m.body_mass[:, body], m0 + added, atol=1e-5)
    centre = torch.stack([x, torch.zeros_like(x), HOOD_TOP_Z + h / 2], -1)
    want = (m0[:, None] * c0 + added[:, None] * centre) / (m0 + added)[:, None]
    assert torch.allclose(m.body_ipos[:, body], want, atol=1e-5)
    assert float((m.body_ipos[:, body] - c0)[:, 0].max()) > 0.005            # forward by centimetres, the base is light
    assert torch.allclose(m.body_subtreemass[:, body], tree0 + added, atol=1e-5)     # the derived constants follow
    # the tensor about the new center: the cylinder lifts every axis, the most about the pitch and yaw axes
    from mjlab.utils.lab_api.math import matrix_from_quat
    rot = matrix_from_quat(m.body_iquat[:, body])
    tensor = rot @ torch.diag_embed(m.body_inertia[:, body]) @ rot.transpose(1, 2)
    assert float(tensor[added.argmax(), 1, 1]) > 4.4e-4 + 1e-5
    assert float((m.body_inertia[:, body].sum(-1)).max() - (m.body_inertia[:, body].sum(-1)).min()) > 1e-5
    env.close()


def test_a_fixed_attachment_changes_no_other_draw():
    n = 16
    bare = _env(n, seed=7)
    body = _ids(bare)
    m0 = bare.sim.model.body_mass[:, body].clone()
    bare.reset()
    quat0 = bare.scene["robot"].data.root_link_quat_w.clone()
    bare.close()
    env = _env(n, seed=7, attachment=Attachment.parse("35,18.5,0.0445,50"))
    assert torch.allclose(env.sim.model.body_mass[:, body], m0 + 0.050, atol=1e-5)
    assert torch.allclose(env.attachment_draw["height_m"], torch.full((n,), 0.035, device="cuda:0"))
    env.reset()
    assert torch.allclose(env.scene["robot"].data.root_link_quat_w, quat0, atol=1e-5)      # the same random numbers
    env.close()


def _start(env, tilts):
    """Each env's start: a slice's name, or a roll in degrees."""
    from mp4_rl.getup.task import SLICES
    n, dev = env.num_envs, env.device
    term = env.event_manager.get_term_cfg("getup_start").func
    exact = torch.full((n, 2), float("nan"), device=dev)
    fixed = torch.full((n,), -1, dtype=torch.long, device=dev)
    for i, t in enumerate(tilts):
        if t in SLICES:
            fixed[i] = SLICES.index(t)
        else:
            exact[i] = torch.tensor((math.radians(t), 0.0), device=dev)
    term._exact, term._fixed_random, term.fixed_slice = exact, torch.zeros(n, dtype=torch.bool, device=dev), fixed
    return term


def _rollover(env):
    rm = env.reward_manager
    return rm._step_reward[:, rm.active_terms.index("rollover")]


def test_the_rollover_cost_charges_a_fall_past_130_and_never_a_back_start():
    from mp4_rl.getup.task import tilt_deg
    starts = ["midfall", "midfall", "crouch", 90.0, 99.0, 110.0, 150.0, 180.0]
    n = len(starts)
    env = _env(n, rollover_cost=2.0, start_joints="stand", fixed_iter=0, spawn_z=(0.2, 0.2))
    _start(env, starts)
    env.reset()
    tilt0 = tilt_deg(env).clone()
    assert float(tilt0[:2].max()) < 75.0 and abs(float(tilt0[5]) - 110.0) < 1 and abs(float(tilt0[7]) - 180.0) < 1
    robot = env.scene["robot"]
    zero = torch.zeros(n, 12, device="cuda:0")

    def place(deg, z=0.3):
        half = math.radians(deg) / 2
        quat = torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0], device="cuda:0").expand(n, 4)
        pos = env.scene.env_origins.clone()
        pos[:, 2] += z
        robot.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(n, 6, device="cuda:0")], -1))

    place(120.0)                                    # below the line nobody pays
    env.step(zero)
    assert float(_rollover(env).abs().max()) == 0.0
    place(170.0)                                    # past it: the starts below 100 deg pay 2 per second, the others nothing
    env.step(zero)
    assert _rollover(env).tolist() == [-2.0] * 5 + [0.0] * 3
    env.close()


def test_the_rollover_cost_follows_each_episodes_own_start():
    env = _env(2, rollover_cost=3.0, start_joints="stand", fixed_iter=0)
    rm = env.reward_manager
    term = rm._term_cfgs[rm.active_terms.index("rollover")].func
    _start(env, ["midfall", 180.0])
    env.reset()
    assert term.charged.tolist() == [True, False]
    _start(env, [180.0, "midfall"])
    env.reset()
    assert term.charged.tolist() == [False, True]
    env.close()


def test_the_midfall_rotation_range_is_the_options():
    from mp4_rl.getup.task import SLICES
    n = 64
    env = _env(n, fixed_iter=0, midfall_omega=(10.0, 13.0))
    term = env.event_manager.get_term_cfg("getup_start").func
    term.fixed_slice = torch.full((n,), SLICES.index("midfall"), dtype=torch.long, device="cuda:0")
    env.reset()
    w = env.scene["robot"].data.root_link_ang_vel_b.norm(dim=-1)
    assert float(w.min()) >= 10.0 - 1e-3 and float(w.max()) <= 13.0 + 1e-3 and float(w.max() - w.min()) > 0.5
    env.close()
