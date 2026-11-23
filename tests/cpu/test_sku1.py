"""The SKU1 (``--robot sku1``) through the skills: export for the SKU1, the evaluators' plant flags, the task options and
envs (masses, CoM height, display box, leg contacts, the height thresholds), the recipes' command lines, the limits
installer; and that none of it changes a SKU2 build."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mp4_sim import paths as sim_paths
from mp4_sim import robots

from mp4_rl.common import bounds as L

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def _meta(d: Path) -> dict:
    return json.loads(str(np.load(d / "policy.npz")["meta"]))


def _export(*args, cwd=REPO):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    import mp4_sim
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(Path(mp4_sim.__file__).resolve().parents[1])])
    r = subprocess.run([PY, "-B", "-m", "mp4_rl.common.export", *map(str, args)], capture_output=True, text=True,
                       cwd=cwd, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_the_sku2_walker_exported_for_the_sku1(tmp_path):
    src = REPO / "mp4_rl/skills/walker/policies/sku2"
    out = tmp_path / "walker_sku1"
    stdout = _export("--from-policy", src, out, "--robot", "sku1")
    assert "exporting for robot sku1: trained on sku2" in stdout
    a, b = np.load(src / "policy.npz"), np.load(out / "policy.npz")
    for k in a.files:
        if k != "meta":
            assert np.array_equal(a[k], b[k]), k                  # the same network
    m = _meta(out)
    assert m["robot"] == "sku1" and m["legs"]["robot"] == "sku1"
    assert m["legs"]["model"] == "robot/assets/sku_1/pupper.xml"
    assert m["contract_sha256"] == sim_paths.sha256(robots.SKU1.contract_json)
    assert m["bounds"]["info"]["limits_sha256"] == sim_paths.sha256(robots.SKU1.limits_json)
    assert m["stand_offset_abd_deg"] == 12.0
    for f in ("sku1_contract.json", "sku1_joint_limits.json"):
        assert (out / f).read_bytes() == (robots.SKU1.measured / f).read_bytes()
    assert not (out / "sku2_contract.json").exists()


def test_the_get_up_start_exported_for_the_sku1_with_its_traced_limits(tmp_path):
    out = tmp_path / "getup_sku1"
    _export(REPO / "mp4_rl/skills/getup/checkpoints/sku2/getup_start/model_6544.pt", out, "--robot", "sku1", "--limits")
    m = _meta(out)
    assert m["robot"] == "sku1" and m["kind"] == "getup" and m["limits"]
    assert m["bounds"]["info"]["hip_envelope"] == "sku1_hip_envelope.json"
    assert m["bounds"]["info"]["limits_sha256"] == sim_paths.sha256(robots.SKU1.limits_json)


def test_eval_flags_read_the_robot_from_the_meta_or_the_flags():
    assert L.eval_legs({}) == {"legs": "sku_2", "foot": "sphere"}
    assert L.eval_legs({}, robot="sku1")["robot"] == "sku1"
    assert L.eval_legs({"robot": "sku1"})["robot"] == "sku1"
    assert L.eval_legs({"legs": {"legs": "sku_2", "robot": "sku1"}})["robot"] == "sku1"
    assert L.eval_legs({"legs": {"legs": "sku_2", "robot": "sku1"}}, robot="sku2") == {"legs": "sku_2", "foot": "sphere"}
    assert L.eval_legs({"display_collision": True}, robot="sku1")["display_collision"] is True
    assert "display_collision" not in L.eval_legs({"display_collision": True}, display_collision=False)
    assert L.eval_legs({"leg_leg_contacts": True})["leg_leg_contacts"] is True
    import argparse
    ap = argparse.ArgumentParser()
    L.add_eval_args(ap)
    assert L.legs_flags(ap.parse_args([])) == {"legs": None, "foot": None, "foot_radius_m": None}
    f = L.legs_flags(ap.parse_args(["--robot", "sku1", "--display-collision"]))
    assert f["robot"] == "sku1" and f["display_collision"] is True
    assert L.eval_plant({}, **f)["robot"] == "sku1"


def test_a_provisional_limits_file_is_refused_for_a_get_up():
    j = L.load(robot="sku1")
    L.refuse_provisional("getup", robot="sku1")                  # the shipped SKU1 limits: traced on the robot
    with pytest.raises(ValueError, match="provisional"):
        L.refuse_provisional("getup", j={**j, "provisional": "a stand-in"})
    L.refuse_provisional("walk", j={**j, "provisional": "a stand-in"})


def test_install_limits_for_the_sku1(tmp_path):
    sys.path.insert(0, str(REPO / "tools"))
    import install_limits
    sha = install_limits.install(robots.SKU1.limits_json, robots.SKU1.hip_envelope_json, tmp_path, "sku1")
    assert sha == sim_paths.sha256(robots.SKU1.limits_json)
    assert (tmp_path / "sku1_joint_limits.sha256").read_text().split() == [sha, "sku1_joint_limits.json"]
    assert (tmp_path / "sku1_hip_envelope.json").read_bytes() == robots.SKU1.hip_envelope_json.read_bytes()


# ------------------------------------------------------------------------------------------------ envs (mjlab)
def test_the_task_options_and_envs_on_the_sku1():
    pytest.importorskip("mjlab")
    import mujoco
    from mp4_sim.envs.attachment import Attachment
    from mp4_sim.robot.model import Legs, robot_z_shift

    from mp4_rl.common.env import TerrainOptions, describe, make_terrain_env_cfg
    from mp4_rl.skills.getup.task import GetupOptions, make_getup_env_cfg
    from mp4_rl.skills.walker.task import WalkerOptions, make_walker_env_cfg
    assert TerrainOptions().masses_json == "sku2_masses.json" and TerrainOptions().robot == "sku2"
    assert TerrainOptions(robot="sku1").masses_json == "sku1_masses_413g.json"
    assert GetupOptions(robot="sku1").masses_json == "sku1_masses_413g.json"
    # SKU2: no robot on the servo action, no z shift event
    c2 = make_terrain_env_cfg(TerrainOptions(num_envs=2))
    assert c2.actions["servo"].robot is None
    # SKU1 walker: its calibration in the decoder, the scan's reference moved, leg contacts, CoM height last
    lidar = Attachment((35.0, 35.0), (47.2, 47.2), (-20.0, 10.0), 19.0, present=0.5, y_mm=(-10.0, 10.0))
    w = WalkerOptions(num_envs=2, robot="sku1", limits=True, stand_offset_abd_deg=12.0, leg_leg_contacts=True,
                      com_height_mm=(61.0, 70.0), stair_exit=True, attachment=lidar)
    cw = make_walker_env_cfg(w)
    assert cw.actions["servo"].robot == "sku1"
    assert list(cw.events)[-2:] == ["attachment", "com_height"]
    shift = robot_z_shift(Legs(robot="sku1"))
    assert shift < 0 and cw.observations["critic"].terms["body_scan"].params["stand_z"] == pytest.approx(0.0725 + shift)
    m = cw.scene.entities["robot"].spec_fn().compile()
    calf = m.geom("FR_calf_collision")
    from mp4_sim.envs.contact import LEG_BIT
    assert int(calf.contype[0]) & LEG_BIT                          # leg <-> leg contacts on
    assert describe(w)["com_height"]["com_height_mm"] == [61.0, 70.0]
    # SKU1 get-up: the display box, the height thresholds moved by the robot's stand
    g = GetupOptions(num_envs=2, robot="sku1", limits=True, display_collision=True, band_deg=(90, 180, 180))
    cg = make_getup_env_cfg(g)
    assert cg.actions["servo"].robot == "sku1"
    assert cg.events["robot_z_shift"].params["shift"] == pytest.approx(shift)
    mg = cg.scene.entities["robot"].spec_fn().compile()
    assert mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_GEOM, "display_collision") >= 0
    with pytest.raises(ValueError, match="leg_leg_contacts"):
        make_getup_env_cfg(GetupOptions(num_envs=2, robot="sku1", leg_leg_contacts=True))


# ------------------------------------------------------------------------------------------------ the recipes
def _recipe(tmp_path, rel, *args, **env):
    dump = tmp_path / "argv"
    stub = tmp_path / "py"
    stub.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done >> {dump}\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    r = subprocess.run(["sh", str(REPO / rel), *args], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHON": str(stub), **env})
    return r, dump.read_text().splitlines()


def test_the_sku1_get_up_recipe(tmp_path):
    pytest.importorskip("mjlab")
    from mp4_rl.skills.getup import train as tg
    r, argv = _recipe(tmp_path, "mp4_rl/skills/getup/recipes/sku1/finetune.sh")
    assert r.returncode == 0, r.stderr
    a = tg.parse_args(argv[argv.index("mp4_rl.skills.getup.train") + 1:])
    assert (a.seed, a.iters, a.resume) == (71, 1000, "mp4_rl/skills/getup/checkpoints/sku2/getup_start/model_6544.pt")
    assert a.robot == "sku1" and a.limits and a.display_collision and a.com_height_mm == [61.0, 70.0]
    o = tg.task_options(a)
    assert o.robot == "sku1" and o.display_collision and o.masses_json == "sku1_masses_413g.json"
    assert o.attachment.present == 0.5 and o.attachment.y_mm == (-10.0, 10.0) and o.attachment.mass_g == (47.2, 47.2)
    L.refuse_provisional("getup", robot=a.robot)                  # refuses nothing: the limits are traced
    assert L.limits_contract_check(robot=a.robot)["match"]


def test_the_sku1_walker_recipe_exports_its_anchor_first(tmp_path):
    pytest.importorskip("mjlab")
    from mp4_rl.skills.walker import train as tw
    out = tmp_path / "out"
    r, argv = _recipe(tmp_path, "mp4_rl/skills/walker/recipes/sku1/finetune.sh", OUT=str(out))
    assert r.returncode == 0, r.stderr
    # first the anchor's export (the stub records it), then the trainer
    assert argv[:4] == ["-B", "-m", "mp4_rl.common.export", "--from-policy"]
    assert argv[4:7] == ["mp4_rl/skills/walker/policies/sku2", f"{out}/anchor_sku1", "--robot"]
    tr = argv[argv.index("mp4_rl.skills.walker.train") + 1:]
    a = tw.build_parser().parse_args(tr)
    assert (a.seed, a.iters, a.resume) == (210, 1000, "mp4_rl/skills/walker/checkpoints/sku2/walker/model_10793.pt")
    assert a.terrain == "rough" and a.anchor_range == [0.2, 0.2, 2.0, 1.0] and a.beta0 == 10.0
    assert a.anchor == f"{out}/anchor_sku1" and a.robot == "sku1" and a.limits and a.leg_leg_contacts
    o = tw.walker_options(a, {"action_scale": [0.25, 0.5, 0.5] * 4, "history": 10}, 10793 * 24)
    assert o.robot == "sku1" and o.leg_leg_contacts and o.com_height_mm == (61.0, 70.0)
    assert o.attachment.x_mm == (-20.0, 10.0) and o.attachment.mass_g == (47.2, 47.2)
    assert o.masses_json == "sku1_masses_413g.json" and o.stand_offset_abd_deg == 12.0
    # with ANCHOR given, no export
    other = tmp_path / "given"
    other.mkdir()
    r, argv = _recipe(other, "mp4_rl/skills/walker/recipes/sku1/finetune.sh", OUT=str(out), ANCHOR="x/anchor")
    assert r.returncode == 0 and "mp4_rl.common.export" not in argv and argv[argv.index("--anchor") + 1] == "x/anchor"


# ------------------------------------------------------------------------------------------------ the classic gait
def test_the_sku1_classic_gait_baseline_runs_on_the_sku1(tmp_path):
    """The SKU1 evaluation script's classic-gait jobs name the robot (the classic has no policy meta to carry it), so
    the plant, its masses and the classic's clamps are the SKU1's."""
    pytest.importorskip("mjlab")
    from mp4_sim.robot.model import masses_path

    from mp4_rl.common.classic_gait import ClassicPolicy
    from mp4_rl.skills.walker.eval import omni
    out = tmp_path / "out"
    r, argv = _recipe(tmp_path, "mp4_rl/skills/walker/eval/sku1.sh", str(out), "classic_s21", WALKER="x")
    assert r.returncode == 0, r.stderr
    args = omni.make_parser().parse_args(argv[argv.index("mp4_rl.skills.walker.eval.omni") + 1:])
    assert args.classic and args.robot == "sku1"
    plant = L.eval_plant(ClassicPolicy("cpu").meta, **L.eval_flags(args))
    assert plant["robot"] == "sku1"
    assert masses_path(args.masses_json, plant["robot"]).is_file()


def test_the_classic_driver_clamps_with_its_robots_limits():
    """``ClassicDriver``'s bounds come from the servo action term's robot (its calibration and its joint limits)."""
    from types import SimpleNamespace

    from mp4_rl.common.classic_gait import ClassicDriver
    from mp4_rl.common.mdp import TorchContract
    for robot in ("sku1", None):
        term = SimpleNamespace(tc=TorchContract("cpu", 0.0, robot), cfg=SimpleNamespace(robot=robot), _shifted=False,
                               reset=lambda ids: None)
        env = SimpleNamespace(device="cpu", num_envs=2,
                              action_manager=SimpleNamespace(get_term=lambda name, t=term: t))
        d = ClassicDriver(env)
        want = L.decoder_bounds("walk", robot=robot)
        assert d.bounds_meta.info["limits_sha256"] == sim_paths.sha256(robots.get(robot).limits_json)
        np.testing.assert_array_equal(d.bounds_meta.lo, want.lo)
        np.testing.assert_array_equal(d.bounds_meta.hi, want.hi)


def test_the_noise_follows_the_robot(monkeypatch):
    """The options' noise defaults to the robot's own measured file (``mp4_sim.robots.Robot.noise_json``)."""
    pytest.importorskip("mjlab")
    from mp4_rl.common import env as E
    asked = []
    monkeypatch.setattr(E, "noise_profile", lambda name="measured", robot=None: asked.append(robot) or {"pos": [robot]})
    assert E.TerrainOptions(robot="sku1").noise_end == {"pos": ["sku1"]}
    assert E.TerrainOptions().noise_end == {"pos": ["sku2"]}
    assert E.TaskOptions().noise_end == {"pos": [None]}
    assert E.TerrainOptions(robot="sku1", noise_end={"pos": [1.0]}).noise_end == {"pos": [1.0]}
    assert asked == ["sku1", "sku2", None]
