"""The robot model's legs and feet (``mp4_sim.robot.model.Legs``): the vendored model, its feet and its stand, the compiled models
pinned."""
import hashlib
import math

import mujoco
import numpy as np
import pytest
from mp4_sim import robots as sim_robots
from mp4_sim.robot import model as sku2
from mp4_sim.robot.model import Legs, legs_meta, legs_of

from mp4_rl import paths
from mp4_rl.common.bounds import eval_legs
from mp4_rl.common.policy import Contract

FEET = (Legs(foot="sphere"), Legs(foot="mesh"))
WALKER = dict(masses_json="sku2_masses.json", self_collision=True, foot_solref=(0.01, 1.0))
# compiled-model hashes (MuJoCo 3.7.0), per foot: the walker's and the get-up's plants, with and without the limits
HASHES = {
    "sphere": {"walker": "ae6537d38238a91a505ec45aa01d853cf4c0ed1c1b20084b7cc94cdcdc302748",
               "walker_limits": "3a47e1b8ffbd60ddb60f797b86ca23b526f3c1f95eef8cd879f15d8d0830e3a2",
               "getup": "7e53f1f4263cb98ce0fc3073da104697bcffab94ef2722b1da90dbbe3052c0fd"},
    "mesh": {"walker": "c0e2d94344c01a50df68d5eea03718c77728bdbe681c9247ad68ca7ce6b14bc8",
             "walker_limits": "f70da4f02220b15981830aa58dc39fcbbe80771ad1707221bf2cce3ab35b5cfa",
             "getup": "d2c68bce61128b6b1f8089fe0db78c9ba8956465f6d8ba413b8b8019211c5b9f"},
}
MODEL_SHA256 = "455316472c9c059c7bb9e2d5236f87c59a6d58e42071c89507fcbe8bf83f3dcf"     # robot/assets/sku_2/pupper.xml


def model_hash(m: mujoco.MjModel) -> str:
    h = hashlib.sha256()
    for k in sorted(dir(m)):
        if k.startswith("_"):
            continue
        try:
            v = getattr(m, k)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


@pytest.mark.skipif(mujoco.__version__ != "3.7.0", reason="hashes pinned with MuJoCo 3.7.0")
@pytest.mark.parametrize("legs", FEET, ids=("sphere", "mesh"))
def test_compiled_models_are_pinned(legs):
    from mp4_rl.getup.task import CONTACT_SOLREF, THIGH_SHORT_M, _spec_fn
    want = HASHES[legs.foot]
    assert model_hash(sku2.get_spec(**WALKER, legs=legs).compile()) == want["walker"]
    assert model_hash(sku2.get_spec(**WALKER, limits=True, legs=legs).compile()) == want["walker_limits"]
    g = _spec_fn(lambda: sku2.get_spec(**WALKER, legs=legs), True, True, THIGH_SHORT_M, CONTACT_SOLREF)
    assert model_hash(g().compile()) == want["getup"]


def test_the_simulation_model_is_the_one_the_policies_were_trained_on():
    xml = sim_robots.SKU2.xml
    assert paths.sha256(xml) == MODEL_SHA256
    spec = mujoco.MjSpec.from_file(str(xml))
    parts = xml.parent / "parts"
    for mesh in spec.meshes:
        assert (parts / mesh.file).is_file(), mesh.name
    for legs in FEET:
        for strip in (True, False):
            sku2.get_spec(strip_visuals=strip, legs=legs).compile()
    assert sku2.sources() == {"mp4_sim:robot/assets/sku_2/pupper.xml": MODEL_SHA256,
                              "mp4_sim:robot/assets/sku_2/measured/sku2_masses.json":
                                  paths.sha256(sim_robots.SKU2.masses_path)}


def _table(m):
    out = {}
    for i in range(m.nbody):
        b = m.body(i)
        out["body", b.name] = (b.mass.tolist(), b.ipos.tolist(), b.inertia.tolist(), b.iquat.tolist(), b.pos.tolist(),
                               b.quat.tolist())
    for i in range(m.njnt):
        j = m.joint(i)
        out["joint", j.name] = (j.type.tolist(), j.axis.tolist(), j.pos.tolist(), j.range.tolist(), j.limited.tolist(),
                                j.damping.tolist(), j.armature.tolist(), j.frictionloss.tolist())
    for i in range(m.ntendon):
        a, n = m.tendon_adr[i], m.tendon_num[i]
        out["tendon", m.tendon(i).name] = (m.wrap_prm[a:a + n].tolist(), m.wrap_objid[a:a + n].tolist(),
                                           m.tendon_damping[i].tolist(), float(m.tendon_armature[i]))
    for i in range(m.ngeom):
        g = m.geom(i)
        out["geom", g.name or f"{m.body(int(g.bodyid[0])).name}#{i}"] = (g.type.tolist(), g.size.tolist(), g.pos.tolist(), g.quat.tolist(), g.contype.tolist(),
                               g.conaffinity.tolist(), g.priority.tolist(), g.solref.tolist(), g.friction.tolist())
    for i in range(m.nsite):
        out["site", m.site(i).name] = m.site(i).pos.tolist()
    return out


def test_the_feet_differ_only_in_the_foot_geoms():
    from mp4_rl.getup.task import getup_collisions
    a, b = (sku2.get_spec(**WALKER, limits=True, legs=legs) for legs in FEET)
    getup_collisions(a)
    getup_collisions(b)
    ta, tb = _table(a.compile()), _table(b.compile())
    assert set(ta) == set(tb)
    assert {k for k in ta if ta[k] != tb[k]} == {("geom", f"{leg}_foot") for leg in sku2.LEGS}
    for leg in sku2.LEGS:          # collision bits, priority, contact parameters: the same
        assert ta["geom", f"{leg}_foot"][4:] == tb["geom", f"{leg}_foot"][4:]
        assert tb["site", f"{leg}_foot"] == pytest.approx([0.0422496, 0.0422496, 0.0])


def test_mesh_feet_and_the_sphere_bottom():
    f = sku2.mesh_feet()
    for leg in sku2.LEGS:
        ext = (f[leg]["bbox_max"] - f[leg]["bbox_min"]) * 1000
        assert ext == pytest.approx([19.26, 19.64, 6.0], abs=0.01)
        assert f[leg]["vertices"] == 2818 and f[leg]["faces"] == 5632
    sphere, mesh = (sku2.stand_geometry(L) for L in FEET)
    assert sphere["base_z_m"] == pytest.approx(mesh["base_z_m"], abs=1e-9)
    for leg in sku2.LEGS:
        assert sphere["foot_site_clearance_m"][leg] == pytest.approx(mesh["foot_site_clearance_m"][leg], abs=1e-9)
        assert sphere["foot_lowest_z_m"][leg] == pytest.approx(0.0, abs=1e-7)
    # a smaller sphere moves its centre, not its bottom
    assert sku2.stand_geometry(Legs(foot_radius_m=0.003))["base_z_m"] == pytest.approx(sphere["base_z_m"], abs=1e-9)


def test_stand_heights():
    g = sku2.stand_geometry()
    assert g["base_z_m"] == pytest.approx(0.075162, abs=2e-6)
    assert all(v == pytest.approx(0.072206, abs=2e-6) for v in g["hip_axis_height_m"].values())
    assert sku2.init_base_z() == pytest.approx(0.0747065, abs=1e-7)
    assert sku2.foot_site_clearance() == pytest.approx(0.00411, abs=1e-5)


def _stand_sim(legs, seconds=2.0):
    """The walker's plant on a floor, position servos (the model's: kp 8, kv 0.05, 0.8 N m) holding the contract's
    stand from the spawn height."""
    c = Contract.load()
    stand = dict(zip(c.joint_order, c.stand_joint_rad))
    spec = sku2.get_spec(**WALKER, legs=legs)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], contype=1, conaffinity=1)
    for leg in sku2.LEGS:
        for name, trn, target in ((f"{leg}_hip", mujoco.mjtTrn.mjTRN_JOINT, f"{leg}_hip"),
                                  (f"{leg}_thigh", mujoco.mjtTrn.mjTRN_JOINT, f"{leg}_thigh"),
                                  (f"{leg}_calf", mujoco.mjtTrn.mjTRN_TENDON, f"{leg}_knee_drive")):
            a = spec.add_actuator(name=name, target=target, trntype=trn)
            a.set_to_position(kp=8.0, kv=0.05)
            a.forcerange, a.forcelimited = [-0.8, 0.8], 1
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2], d.qpos[3] = sku2.init_base_z(legs), 1.0
    for name, v in stand.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    mujoco.mj_forward(m, d)
    for leg in sku2.LEGS:
        d.ctrl[m.actuator(f"{leg}_hip").id] = stand[f"{leg}_hip"]
        d.ctrl[m.actuator(f"{leg}_thigh").id] = stand[f"{leg}_thigh"]
        d.ctrl[m.actuator(f"{leg}_calf").id] = d.ten_length[m.tendon(f"{leg}_knee_drive").id]
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    tilt = math.degrees(math.acos(min(1.0, d.xmat[m.body("base").id][8])))
    return float(d.qpos[2]), tilt, d


@pytest.mark.parametrize("legs", FEET, ids=("sphere", "mesh"))
def test_stand_for_2_s_on_flat_ground(legs):
    z, tilt, d = _stand_sim(legs)
    assert tilt < 0.5
    assert z == pytest.approx(sku2.stand_geometry(legs)["base_z_m"], abs=0.001)     # within 1 mm of the kinematic stand
    assert d.ncon == 4                                                               # the four feet, nothing else


def test_options_meta_and_flags():
    with pytest.raises(ValueError):
        Legs("other")
    with pytest.raises(ValueError):
        Legs(foot="cube")
    with pytest.raises(ValueError):
        Legs(foot_radius_m=0.03)
    assert legs_of(None) == legs_of({}) == legs_of({"legs": None, "foot": None}) == Legs() == Legs("new")
    for legs in FEET + (Legs(foot_radius_m=0.006),):
        meta = {"legs": legs_meta(legs)}
        assert legs_of(meta) == legs
        assert legs_of({**eval_legs(meta)}) == legs
    assert eval_legs({}) == {"legs": "sku_2", "foot": "sphere"}
    assert eval_legs({}, "new") == {"legs": "sku_2", "foot": "sphere"}
    assert eval_legs({"legs": legs_meta(Legs(foot_radius_m=0.006))}) == {"legs": "sku_2", "foot": "sphere",
                                                                          "foot_radius_m": 0.006}
    assert eval_legs({"legs": legs_meta(Legs(foot_radius_m=0.006))}, foot="mesh") == {"legs": "sku_2", "foot": "mesh"}
    m = legs_meta()
    assert (m["legs"], m["model"], m["model_sha256"]) == ("sku_2", "robot/assets/sku_2/pupper.xml", MODEL_SHA256)
