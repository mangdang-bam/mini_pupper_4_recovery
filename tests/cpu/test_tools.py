"""The tools: a new contract, the deployed subset, the summary of logs."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from robot_helpers import go

from mp4_rl import paths
from mp4_rl.common.policy import CONTRACT_JSON, Contract
from mp4_rl.getup.mp4link import cli
from mp4_rl.getup.runner import checks

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def offsets_file(tmp_path, offsets, stored=True):
    ticks = cli.offset_ticks(offsets)
    f = tmp_path / "offsets.json"
    f.write_text(json.dumps({"offsets_deg": [float(v) for v in offsets], "offsets_stored": [stored] * 12,
                             "stand_ticks": ticks.tolist(), "stand_dd": cli.ticks_to_dd(ticks).tolist()}))
    return f


def test_make_contract_for_another_unit(tmp_path, capsys):
    tool = load_tool("make_contract")
    c0 = Contract.load()
    new = c0.offsets_deg.astype(np.float32) + 6.0
    out = tmp_path / "contract.json"
    assert tool.main([str(offsets_file(tmp_path, new)), "--note", "unit 2", "--out", str(out)]) == 0
    c = Contract.load(out)
    d = json.loads(out.read_text())
    assert np.allclose(c.offsets_deg, new) and d["calibration"] == "unit 2"
    assert c.stand_dd.tolist() == cli.ticks_to_dd(cli.offset_ticks(new)).tolist()
    assert d["earlier_calibrations"][-1]["sha256"] == paths.sha256(CONTRACT_JSON)
    for key in ("signs", "tendon_coefs", "stand_policy_rad", "joint_order"):
        assert d[key] == c0.meta[key]
    assert "re-export" in capsys.readouterr().out


def test_make_contract_refuses_unset_offsets_and_a_wrong_stand(tmp_path):
    tool = load_tool("make_contract")
    c0 = Contract.load()
    with pytest.raises(SystemExit, match="not every offset"):
        tool.main([str(offsets_file(tmp_path, c0.offsets_deg, stored=False)), "--note", "x", "--out", str(tmp_path / "c.json")])
    f = offsets_file(tmp_path, c0.offsets_deg)
    d = json.loads(f.read_text())
    d["stand_dd"][0] += 10
    f.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not the one derived"):
        tool.main([str(f), "--note", "x", "--out", str(tmp_path / "c.json")])


def test_a_shipped_policy_is_refused_on_a_new_calibration(tmp_path, monkeypatch):
    tool = load_tool("make_contract")
    c0 = Contract.load()
    out = tmp_path / "contract.json"
    tool.main([str(offsets_file(tmp_path, c0.offsets_deg.astype(np.float32) + 6.0)), "--note", "unit 2", "--out", str(out)])
    meta = json.loads(str(np.load(paths.GETUP_POLICIES / "sku2" / "policy.npz")["meta"]))
    assert checks.band_refusal(meta) is None
    assert "another calibration" in checks.band_refusal(meta, Contract.load(out))


def test_deployed_subset_needs_only_numpy_and_pyserial(tmp_path):
    """The deployed folder runs the runner with numpy, pyserial and two files of mp4_sim (its import-free
    ``__init__.py`` and ``calibration.py``); the policies pass their check there with the calibration they carry."""
    import mp4_sim
    dest = tmp_path / "walk"
    env0 = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHON": sys.executable}
    env0["PYTHONPATH"] = str(Path(mp4_sim.__file__).resolve().parents[1])      # the deploying machine's mp4_sim
    subprocess.run(["sh", str(ROOT / "tools" / "deploy.sh"), "-", str(dest)], check=True, capture_output=True, env=env0)
    assert sorted(p.name for p in (dest / "src" / "mp4_sim").iterdir()) == ["__init__.py", "calibration.py"]
    assert sorted(p.name for p in (dest / "src").iterdir()) == ["mp4_rl", "mp4_sim"]
    assert not (dest / "src" / "mp4_rl" / "walker").exists() and not (dest / "src" / "mp4_rl" / "common" / "env.py").exists()
    mount = dest / "src" / "mp4_rl" / "getup" / "runner" / "imu_mount.json"
    assert mount.read_bytes() == (Path(mp4_sim.__file__).parent / "robot/assets/sku_2/measured/imu_mount.json").read_bytes()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONPATH": str(dest / "src")}
    code = ("import sys; import mp4_rl.getup.runner.runner as r, mp4_rl.getup.mp4link.cli;"
            "std = set(sys.stdlib_module_names);"
            "top = {m.split('.')[0] for m in sys.modules} - std - {'numpy', 'serial', 'mp4_rl', 'mp4_sim'};"
            "assert not {t for t in top if not t.startswith('_')}, top;"
            "sim = sorted(m for m in sys.modules if m.split('.')[0] == 'mp4_sim');"
            "assert sim == ['mp4_sim', 'mp4_sim.calibration'], sim;"
            "files = [getattr(m, '__file__', '') or '' for n, m in sys.modules.items() if n.split('.')[0] in ('mp4_rl', 'mp4_sim')];"
            "assert all(f.startswith(sys.argv[1]) for f in files), files;"
            "assert r.IMU_MOUNT_JSON.parent == r.Path(r.__file__).resolve().parent")
    subprocess.run([sys.executable, "-B", "-c", code, str(dest.resolve())], check=True, env=env, cwd=dest)
    for skill in ("walker", "getup"):
        folder = dest / "skills" / skill / "policies" / "sku2"
        assert sorted(p.name for p in folder.iterdir()) == ["config.json", "policy.npz", "sku2_contract.json",
                                                            "sku2_joint_limits.json"]
        p = subprocess.run([sys.executable, "-B", "-m", "mp4_rl.getup.runner", "check", "--policy",
                            f"skills/{skill}/policies/sku2"], env=env, cwd=dest, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "calibration: ok (sku2_contract.json" in p.stdout


def test_summarise_runs_on_a_fresh_log(robot, tmp_path, capsys):
    from fake_esp32 import fall_and_rise, roll
    robot.script, robot.script_at = roll(fall_and_rise(lie_s=0.5, up_s=0.8)), 60
    rc, _, _ = go(robot, tmp_path, seconds=3.0)
    assert rc == 0
    tool = load_tool("summarise_runs")
    capsys.readouterr()
    assert tool.main([str(tmp_path), "--json", str(tmp_path / "s.json")]) == 0
    assert "fall: 1 get-ups, 1 up, 1 on the first attempt, 0 never up" in capsys.readouterr().out
    row = json.loads((tmp_path / "s.json").read_text())["runs"][0]
    assert row["events"][0]["up"] and row["steps"] == 150


def test_summarise_filters_a_one_frame_temperature_spike():
    tool = load_tool("summarise_runs")
    t = np.full((10, 12), 33.6)
    t[5, 1] = 67.2
    assert tool.median3(t).max() == pytest.approx(33.6)


def fake_run(d, name, tilt, getups, policy="/pi/runs/pol_a"):
    """A ``summarise_runs`` input: a policy log with the tilt per step, its raw log, and ``getups`` in the meta."""
    n = len(tilt)
    meta = {"getups": getups, "getup_policy": policy, "getup_phase_s": 4.0, "getup_phase_restart_s": 1.0}
    np.savez(d / f"{name}.policy.npz", meta=json.dumps(meta), tick=np.arange(n) * 4, tilt_cf_deg=np.asarray(tilt, float),
             t_send=np.arange(n) * 0.02)
    np.savez(d / f"{name}.npz", temp_dc=np.full((4, 12), 300), cur_ma=np.zeros((4, 12), int), sent_t=np.arange(n) * 0.02)


def test_summarise_counts_a_restarted_get_up_the_log_cut_off(tmp_path, capsys):
    tool = load_tool("summarise_runs")
    tilt = np.r_[np.full(100, 50.0), np.full(250, 176.0), np.full(250, 176.0), np.full(150, 176.0)]
    fake_run(tmp_path, "cut", tilt, [{"step": 100, "tick": 400, "gravity": [0, -0.7, -0.7], "phase_restarts": [350, 600]}])
    long_tilt = np.r_[np.full(100, 50.0), np.full(1100, 176.0)]
    fake_run(tmp_path, "stuck", long_tilt, [{"step": 100, "tick": 400, "gravity": [0, 0.7, -0.7], "phase_restarts": [350],
                                             "end": "timeout", "end_step": 1200, "seconds": 22.0}])
    no_retry = np.r_[np.full(100, 50.0), np.full(400, 30.0)]    # low enough never to restart, its phase over
    fake_run(tmp_path, "low", no_retry, [{"step": 100, "tick": 400, "gravity": [0, 0.0, -1.0]}])
    assert tool.main([str(tmp_path), "--json", str(tmp_path / "s.json")]) == 0
    runs = {r["name"]: r["events"][0] for r in json.loads((tmp_path / "s.json").read_text())["runs"]}
    assert runs["cut"]["cut_off"] and runs["cut"]["restarts"] == 2
    assert not runs["stuck"]["cut_off"] and not runs["stuck"]["up"]
    assert runs["low"]["cut_off"] is False and runs["low"]["end"] is None
    assert "fall: 3 get-ups, 0 up, 0 on the first attempt, 2 never up, 1 cut" in capsys.readouterr().out


def test_summarise_judges_the_first_attempt_by_side(tmp_path, capsys):
    """A first attempt that lies at 115 deg and rolls over only after its restart is not over 170 deg."""
    tool = load_tool("summarise_runs")
    tilt = np.r_[np.full(50, 50.0), np.full(250, 115.0), np.full(200, 178.0)]
    fake_run(tmp_path, "a", tilt, [{"step": 50, "tick": 200, "gravity": [0, 0.74, -0.65], "phase_restarts": [300]}])
    tilt = np.r_[np.full(50, 50.0), np.full(50, 175.0), np.full(100, 5.0), np.full(50, 50.0), np.full(60, 90.0), np.full(90, 5.0)]
    fake_run(tmp_path, "b", tilt, [
        {"step": 50, "tick": 200, "gravity": [0, -0.74, -0.65], "end": "hand-back", "end_step": 200, "seconds": 3.0},
        {"step": 200, "tick": 800, "gravity": [0, 0.05, -1.0], "end": "hand-back", "end_step": 310, "seconds": 2.2}])
    assert tool.main([str(tmp_path), "--attempts", "--json", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "falls' first attempts over 170 deg: all 1 of 3, right 1 of 1, left 0 of 1, neither 0 of 1" in out
    assert "5.00 s to 115 deg, 4.00 s to 178 deg" in out
    d = json.loads((tmp_path / "s.json").read_text())
    (a,), (b, c) = (r["events"] for r in d["runs"])
    assert a["side"] == "left" and b["side"] == "right" and c["side"] == "neither"
    assert [round(x["max_tilt_deg"]) for x in a["attempts"]] == [115, 178]
    assert d["first_attempts_over"]["all"] == {"n": 3, "over": 1}


def test_summarise_keeps_one_get_up_policy(tmp_path, capsys):
    tool = load_tool("summarise_runs")
    tilt = np.r_[np.full(50, 50.0), np.full(50, 175.0)]
    fake_run(tmp_path, "a", tilt, [{"step": 50, "tick": 200, "gravity": [0, -0.7, -0.7]}])
    fake_run(tmp_path, "b", tilt, [{"step": 50, "tick": 200, "gravity": [0, -0.7, -0.7]}], policy="/pi/runs/pol_b")
    assert tool.main([str(tmp_path), "--getup-policy", "pol_b"]) == 0
    out = capsys.readouterr().out
    assert "fall: 1 get-ups" in out and out.startswith("b ")
    assert tool.main([str(tmp_path), "--getup-policy", "nothing"]) == 1
