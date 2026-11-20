"""The shipped policies and checkpoints: arrays and files pinned, the numpy forward pass against their fixtures,
metadata free of local paths, the policies what ``common.export`` writes from the shipped checkpoints (arrays and the
whole meta), the calibration and limits copies in each folder, the recipes' command lines."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mp4_sim import limits as sim_limits
from mp4_sim import robots as sim_robots

from mp4_rl import paths
from mp4_rl.common.policy import CONTRACT_JSON, Bounds
from mp4_rl.common.policy_file import NumpyPolicy, fixture_tolerance
from mp4_rl.getup.runner import runner as rn

REPO = Path(__file__).resolve().parents[2]
ARRAYS = {
    "walker": "e301f53fa9cd1985b7984c78b0ed37313d742ca519922b358692965d85ae5efa",
    "getup": "c1eb8ab20780a51e9e9cc19bb54ddd0f3eb1482dbc04103968b2d0ce9a806e00",
}
# the policy files' bytes (they are never written again: a re-export must reproduce the arrays and the meta)
NPZ = {
    "walker": "5f7f2d70864c377c87aef3ae7546ed9def5c7deac862e21b22644c6a083372dd",
    "getup": "6b537110f8924d66a648882b2ddc24b065a39450b718c9731f1ab57b91be5c6d",
}
FINAL = {"walker": "walker/model_10793.pt", "getup": "getup/model_7543.pt"}
# each policy's final checkpoint, and the checkpoints the recipes start from (iteration, an observation normaliser)
CHECKPOINTS = {"walker/model_10793.pt": (10793, False), "getup/model_7543.pt": (7543, True),
               "walker_start/model_6297.pt": (6297, False), "getup_start/model_6544.pt": (6544, True)}


def policy_dir(name: str) -> Path:
    return REPO / "skills" / name / "policies" / "sku2"


def checkpoint(rel: str) -> Path:
    skill = "walker" if rel.startswith("walker") else "getup"
    return REPO / "skills" / skill / "checkpoints" / "sku2" / rel
LOCAL = ("/home", "/tmp", "wandb", "runs/")


def arrays_sha256(path) -> str:
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as d:
        for k in sorted(d.files):
            if k != "meta":
                h.update(k.encode())
                h.update(d[k].tobytes())
    return h.hexdigest()


def same_meta(made: dict, shipped: dict) -> None:
    for k in set(made) | set(shipped):
        if k != "source_policy_sha256":
            assert made.get(k) == shipped.get(k), k


@pytest.mark.parametrize("name", ARRAYS)
def test_arrays_and_fixtures(name):
    path = policy_dir(name) / "policy.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == NPZ[name]
    assert arrays_sha256(path) == ARRAYS[name]
    p = NumpyPolicy(path)
    d = np.load(path, allow_pickle=False)
    assert np.abs(p(d["fixture_obs"]) - d["fixture_act"]).max() < fixture_tolerance(d["fixture_act"])


@pytest.mark.parametrize("name", ARRAYS)
def test_meta_and_config(name):
    meta = NumpyPolicy(policy_dir(name) / "policy.npz").meta
    assert meta["contract_sha256"] == paths.sha256(CONTRACT_JSON)
    assert meta["checkpoint_sha256"] == paths.sha256(checkpoint(FINAL[name]))
    assert meta["bounds"]["info"]["limits_sha256"] == paths.sha256(sim_limits.LIMITS_JSON) == sim_limits.LIMITS_SHA256
    assert meta["legs"]["model_sha256"] == paths.sha256(sim_robots.SKU2.xml)
    Bounds.from_meta(meta["bounds"])
    for text in (json.dumps(meta), (policy_dir(name) / "config.json").read_text()):
        assert not any(s in text for s in LOCAL), text[:200]


@pytest.mark.parametrize("name", ARRAYS)
def test_each_folder_carries_byte_copies_of_the_calibration_and_the_limits(name):
    for f in (sim_robots.SKU2.contract_json, sim_robots.SKU2.limits_json):
        assert (policy_dir(name) / f.name).read_bytes() == f.read_bytes()
    assert sorted(p.name for p in policy_dir(name).iterdir()) == [
        "config.json", "policy.npz", "sku2_contract.json", "sku2_joint_limits.json"]


@pytest.mark.parametrize("name", ARRAYS)
def test_the_configs_name_the_simulation_files_and_their_hashes(name):
    """config.json names the robot's files as mp4_sim names, with the hashes of mp4_sim's files."""
    from mp4_sim import paths as sim_paths
    cfg = json.loads((policy_dir(name) / "config.json").read_text())["task"]
    names = list(cfg["sources"]) + [r["params"] for r in cfg["servo_params"]["roles"].values()] + \
        [cfg["limits"]["limits_json"]]
    assert len(names) == 7 and all(n.startswith("mp4_sim:") for n in names)
    for n, sha in cfg["sources"].items():
        assert sim_paths.sha256(sim_paths.resolve(n)) == sha, n
    for r in cfg["servo_params"]["roles"].values():
        assert sim_paths.sha256(sim_paths.resolve(r["params"])) == r["sha256"]


@pytest.mark.parametrize("name", ARRAYS)
def test_runner_check_passes(name):
    assert rn.main(["check", "--policy", str(policy_dir(name))]) == 0


def test_the_get_up_holds_the_packaged_hip_envelope():
    from mp4_rl.common import bounds as limits
    meta = NumpyPolicy(policy_dir("getup") / "policy.npz").meta
    assert meta["bounds"]["hip_envelope"] == limits.hip_envelope()


def test_the_get_up_stops_the_front_right_hip_at_its_limit():
    """No thigh stop: at straight abduction the FR thigh's backward side is open (the track clamp acts), and where the
    hip envelope caps it (a traced chassis contact) the target stops there."""
    from mp4_rl.common.policy import Contract
    c = Contract.load()
    st = c.stand_rel_rad
    b = Bounds.from_meta(NumpyPolicy(policy_dir("getup") / "policy.npz").meta["bounds"])
    e = b.hip_envelope["FR"]
    rel = np.zeros(12)
    rel[1] = np.radians(90.0)                          # the model's + is backward on this thigh
    assert np.interp(st[0], e["abd"], e["hi"]) >= 10.0 - 1e-9                       # open at the stand's abduction
    assert b.clamp_rel(rel, st)[1] == pytest.approx(b.hi[1] - st[1], abs=1e-12)     # the track clamp
    capped = [(a, h) for a, h in zip(e["abd"], e["hi"]) if h < 10.0 and b.lo[0] <= a <= b.hi[0]]
    assert capped                                       # some columns are capped at a traced contact
    for a, h in capped:
        rel[0] = a - st[0]
        assert b.clamp_rel(rel, st)[1] == pytest.approx(min(h, b.hi[1]) - st[1], abs=1e-9), np.degrees(a)


def export(*args):
    import mp4_sim
    src = os.pathsep.join([str(REPO / "src"), str(Path(mp4_sim.__file__).resolve().parents[1])])
    subprocess.run([sys.executable, "-m", "mp4_rl.common.export", *map(str, args)], check=True, cwd=REPO,
                   env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin", "WANDB_MODE": "disabled"})


@pytest.mark.parametrize("name", ARRAYS)
def test_export_from_policy_keeps_arrays_and_meta(tmp_path, name):
    out = tmp_path / name
    export("--from-policy", policy_dir(name), out)
    assert arrays_sha256(out / "policy.npz") == ARRAYS[name]
    made, shipped = (NumpyPolicy(p / "policy.npz").meta for p in (out, policy_dir(name)))
    assert str(tmp_path) not in json.dumps(made) and str(REPO) not in json.dumps(made)
    same_meta(made, shipped)
    assert made["source_policy_sha256"] == NPZ[name]
    for f in ("config.json", "sku2_contract.json", "sku2_joint_limits.json"):
        assert (out / f).read_bytes() == (policy_dir(name) / f).read_bytes(), f


@pytest.mark.parametrize("name", ARRAYS)
def test_the_final_checkpoints_export_to_the_shipped_policies(tmp_path, name):
    out = tmp_path / name
    export(checkpoint(FINAL[name]), out)
    assert arrays_sha256(out / "policy.npz") == ARRAYS[name]
    same_meta(NumpyPolicy(out / "policy.npz").meta, NumpyPolicy(policy_dir(name) / "policy.npz").meta)
    for f in ("sku2_contract.json", "sku2_joint_limits.json"):
        assert (out / f).read_bytes() == (policy_dir(name) / f).read_bytes(), f


def test_checkpoints_load_as_plain_tensors_with_their_options():
    import torch
    for name, (it, norm) in CHECKPOINTS.items():
        ck = torch.load(checkpoint(name), map_location="cpu", weights_only=True)
        assert ck.get("iter") == it and ck["critic_state_dict"] and ck["optimizer_state_dict"]
        assert ("obs_normalizer._mean" in ck["actor_state_dict"]) == norm
        cfg = checkpoint(name).parent / "config.json"
        o = json.loads(cfg.read_text())["task"]["options"]
        assert len(o["action_scale"]) == 12 and o["kind"] in ("walk", "getup")
        assert not any(s in cfg.read_text() for s in LOCAL)


def recipe(tmp_path, name, *args, cwd=REPO, **env):
    """Run a recipe with a ``python`` that records its arguments; returns (exit code, the arguments as a dict, stderr)."""
    stub = tmp_path / "python"
    stub.write_text(f'#!/bin/sh\necho "$@" > {tmp_path / "args"}\n')
    stub.chmod(0o755)
    r = subprocess.run(["sh", str(REPO / name), *args], cwd=cwd, capture_output=True, text=True,
                       env={"PYTHON": str(stub), "PATH": "/usr/bin:/bin", **env})
    a = (tmp_path / "args").read_text().split() if (tmp_path / "args").exists() else []
    opts = {}
    for i, x in enumerate(a):
        if x.startswith("--"):
            opts[x] = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else True
    return r.returncode, opts, r.stderr


@pytest.mark.parametrize("name, seed, iters, start, anchor", [
    ("skills/walker/recipes/sku2/step1.sh", "71", "1000", "skills/walker/checkpoints/sku2/walker_start/model_6297.pt", "skills/walker/checkpoints/sku2/walker_start/model_6297.pt"),
    ("skills/getup/recipes/sku2/finetune.sh", "72", "1000", "skills/getup/checkpoints/sku2/getup_start/model_6544.pt", None),
])
def test_recipes_start_from_the_shipped_checkpoints(tmp_path, name, seed, iters, start, anchor):
    rc, o, err = recipe(tmp_path, name)
    assert rc == 0, err
    assert (o["--seed"], o["--iters"], o["--resume"], o.get("--anchor")) == (seed, iters, start, anchor)
    assert (REPO / start).is_file() and (REPO / start).with_name("config.json").is_file()
    assert o["--actuator"] == "md300" and "--masses-json" not in o and "--legs" not in o      # the packaged robot


def test_recipe_settings_and_overrides(tmp_path):
    rc, o, _ = recipe(tmp_path, "skills/walker/recipes/sku2/step1.sh", "5", "my/parent.pt", "2", ENVS="64", OUT="runs/smoke")
    assert rc == 0 and (o["--seed"], o["--resume"], o["--anchor"], o["--iters"]) == ("5", "my/parent.pt", "my/parent.pt", "2")
    assert (o["--envs"], o["--out"], o["--cmd-ramp"], o["--save-at"]) == ("64", "runs/smoke", "300", "300")
    rc, o, _ = recipe(tmp_path, "skills/getup/recipes/sku2/finetune.sh", ENVS="64")
    assert rc == 0 and o["--envs"] == "64" and o["--std-bounds"] == "0.06" and o["--getup-attachment"] is True


@pytest.mark.parametrize("script, name, flag, full, quick", [
    ("skills/getup/eval/sku2.sh", "gflat_s21", "--robots", "512", "64"),
    ("skills/getup/eval/sku1.sh", "gterr_s21", "--robots", "1024", "128"),
    ("skills/walker/eval/sku2.sh", "omni_s21", "--per", "24", "3"),
    ("skills/walker/eval/sku1.sh", "push_s21", "--robots", None, "8"),
    ("skills/walker/eval/sku2.sh", "wstairs_s21", "--envs", "256", "32"),
])
def test_quick_evaluations_run_an_eighth_of_the_robots(tmp_path, script, name, flag, full, quick):
    for env, want in (({}, full), ({"QUICK": "1"}, quick)):
        rc, o, err = recipe(tmp_path, script, str(tmp_path / f"out{len(env)}"), name, WALKER="w", GETUP="g", **env)
        assert rc == 0 and o.get(flag) == want, (env, o, err)


@pytest.mark.parametrize("n, seed, iters", [(2, "81", "1500"), (3, "91", "1000"), (4, "101", "1000")])
def test_later_walker_steps_continue_from_the_step_before(tmp_path, n, seed, iters):
    work = tmp_path / "w"
    rc, _, err = recipe(tmp_path, f"skills/walker/recipes/sku2/step{n}.sh", cwd=work.mkdir() or work)
    assert rc == 2 and f"no run in runs/walker_step{n - 1}" in err
    run = work / f"runs/walker_step{n - 1}/20270101_000000_s1_step{n - 1}"
    run.mkdir(parents=True)
    for k in (7050, 7297, 7100):
        (run / f"model_{k}.pt").touch()
    rc, o, err = recipe(tmp_path, f"skills/walker/recipes/sku2/step{n}.sh", cwd=work)
    assert rc == 0, err
    assert (o["--seed"], o["--iters"], o["--tag"]) == (seed, iters, f"step{n}")
    assert o["--resume"] == f"runs/walker_step{n - 1}/{run.name}/model_7297.pt"
    assert o["--anchor"] == "skills/walker/checkpoints/sku2/walker_start/model_6297.pt" and o["--cmd-ramp"] == "0"
