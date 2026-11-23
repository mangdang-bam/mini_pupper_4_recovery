"""eval/sequence.py: its vectorised complementary filter is the runner's, its observation layout is mjlab's (and
``policy.HistoryStacker``'s), and on the GPU a forced GETUP switches the action config, seeds the decoder and hands
back to the walker as the runner does."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp4_rl import paths
from mp4_rl.common.policy import CONTRACT_JSON, HistoryStacker  # noqa: E402
from mp4_rl.deploy.runner import runner as rn  # noqa: E402
from mp4_rl.skills.getup.eval import sequence as sc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WALKER = REPO / "mp4_rl" / "skills" / "walker" / "policies" / "sku2"


def test_torch_filter_is_the_runners():
    rng = np.random.default_rng(0)
    T, n = 300, 3
    phi = np.clip((np.arange(T) * 0.02 - 1.0) * 300.0, 0, 90)          # a roll at 300 deg/s after 1 s
    rate = np.gradient(np.radians(phi), 0.02)
    gyro = np.zeros((T, n, 3))
    acc = np.zeros((T, n, 3))
    gyro[:, 0, 0] = rate
    acc[:, 0] = np.stack([0 * phi, 1000 * np.sin(np.radians(phi)), 1000 * np.cos(np.radians(phi))], -1)
    acc[:, 1] = [0, 0, 1000]
    acc[100:105, 1] = rng.normal(0, 1500, (5, 3))                      # strikes
    gyro[:, 2] = rng.normal(0, 0.3, (T, 3))
    acc[:, 2] = rng.normal([0, 0, 1000], 300, (T, 3))
    acc[50:52, 2] = 0.0                                                # dropouts
    ref = [rn.GravityFilter() for _ in range(n)]
    tf = sc.TorchGravityFilter(n, "cpu")
    for t in range(T):
        got = tf.update(torch.tensor(gyro[t], dtype=torch.float64), torch.tensor(acc[t], dtype=torch.float64), 0.02)
        want = [f.update(gyro[t, i], acc[t, i], 0.02) for i, f in enumerate(ref)]
        assert np.allclose(got.numpy(), want, atol=1e-6, equal_nan=True), (t, got, want)


def test_phase_is_the_runners():
    t = torch.tensor([0.0, 0.02, 0.5, 1.0, 3.99, 4.0, 7.3], dtype=torch.float64)
    got = sc.phase_obs(t, 4.0).numpy()
    want = np.array([rn.getup_phase(float(x), 4.0) for x in t])
    assert np.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("heading,history", [(False, 0), (False, 10), (True, 10), (True, 3)])
def test_layout_matches_the_history_stacker(heading, history):
    rng = np.random.default_rng(1)
    width = 46 if heading else 45
    frames = rng.normal(size=(12, width)).astype(np.float32)
    lay = sc.obs_layout(heading, history)
    stack = HistoryStacker(max(history, 1), heading)
    for f in frames:
        v = torch.tensor(stack(f))[None]
        assert torch.equal(sc.newest_frame(v, lay)[0], torch.tensor(f))
    fresh = HistoryStacker(max(history, 1), heading)
    assert np.array_equal(sc.tiled(torch.tensor(frames[-1])[None], lay)[0].numpy(), fresh(frames[-1]))
    if history:
        ts = sc.TorchStacker(history, heading, 2, "cpu")
        ref = [HistoryStacker(history, heading), HistoryStacker(history, heading)]
        for k, f in enumerate(frames):
            if k == 5:                                                     # robot 1's history cleared
                ts.reset(torch.tensor([1]))
                ref[1].reset()
            got = ts(torch.tensor(np.stack([f, -f])))
            assert np.array_equal(got[0].numpy(), ref[0](f)) and np.array_equal(got[1].numpy(), ref[1](-f))


def zero_getup(d: Path, **meta_extra) -> Path:
    """A get-up policy that always asks for the stand (kind getup, the 90 deg band, 1.57 rad, no history)."""
    d.mkdir(parents=True)
    meta = {"contract_sha256": paths.sha256(CONTRACT_JSON), "kind": "getup", "band_deg": [90.0] * 12, "history": 0,
            "omni": True, **meta_extra}
    np.savez(d / "policy.npz", n_layers=1, scale=np.full(12, 1.57, dtype=np.float32), meta=json.dumps(meta),
             fixture_obs=np.zeros((2, 45), np.float32), fixture_act=np.zeros((2, 12), np.float32),
             W0=np.zeros((12, 45), np.float32), b0=np.zeros(12, np.float32))
    return d


def test_forced_getup_switches_and_hands_back(tmp_path):
    """Four walking robots forced into GETUP at 1 s with a stand-seeking get-up policy: the get-up's scale and band
    apply, its first target is within the 20 deg step of the measured pose, the hand-back comes when the criterion has
    held 25 steps, the walker's config is back and the robots walk again; no robot ends."""
    g = zero_getup(tmp_path / "getup")
    out = tmp_path / "sequence.json"
    rc = sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "4", "--seconds", "5", "--kick", "0",
                  "--force-getup-t", "1.0", "--seed", "11", "--json", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    r, s = d["robots"], d["summary"]
    assert r["getups"] == [1] * 4 and r["end"] == [""] * 4
    ev = d["events"]
    entries = [e["k"] for e in ev if e["event"] == "getup"]
    hbs = [e["k"] for e in ev if e["event"] == "hand-back"]
    assert entries == [50] * 4 and len(hbs) == 4 and all(h - 50 >= rn.HANDBACK_STEPS for h in hbs)
    assert all(0.5 <= t <= 1.5 for t in r["t_up_s"])
    assert s["first_target_err_deg_max"] <= 20.0 + 0.32
    assert s["walking_again"] == 1.0
    assert d["plant"]["plant"] == "getup" and d["getup_band_deg"] == [90.0] * 12




def test_cli_refuses_a_walker_as_getup(tmp_path, capsys):
    """Refused from the policies' meta, before the GPU is touched."""
    rc = sc.main(["--walker", str(WALKER), "--getup", str(WALKER), "--robots", "2", "--json", str(tmp_path / "x")])
    assert rc == 2 and "need 'walk' / 'getup'" in capsys.readouterr().out


def test_kick_schedule_draws():
    """U(LO, HI) magnitudes, U(0, 360) directions, intervals U(A, B) s on the step grid (the first too), only live
    robots kicked but every due robot rescheduled, the same draws for the same seed."""
    n, lo, hi, A, B = 400, 0.5, 1.5, 2.0, 4.0
    runs = []
    for _ in range(2):
        ks = sc.KickSchedule(n, lo, hi, A, B, seed=11, device="cpu")
        first = ks.next_k.clone()
        live = torch.ones(n, dtype=torch.bool)
        live[:10] = False
        log = []
        for k in range(1500):
            ids, m, d = ks.due(k, live)
            log += [(k, int(i), float(mm), float(dd)) for i, mm, dd in zip(ids, m, d)]
        runs.append((first, log))
    (first, log), (first2, log2) = runs
    assert torch.equal(first, first2) and log == log2
    assert int(first.min()) >= 100 and int(first.max()) <= 200
    assert all(r >= 10 for _, r, _, _ in log)                            # dead robots are never kicked
    mags = np.array([x[2] for x in log])
    degs = np.array([x[3] for x in log])
    assert mags.min() >= lo and mags.max() <= hi and abs(mags.mean() - 1.0) < 0.03
    assert degs.min() >= 0 and degs.max() < 360 and abs(degs.mean() - 180) < 8
    for r in range(10, 40):
        kk = [k for k, rr, _, _ in log if rr == r]
        assert kk[0] == int(first[r])
        gaps = np.diff(kk)
        assert len(gaps) >= 5 and gaps.min() >= 100 and gaps.max() <= 200
    ks = sc.KickSchedule(3, 1.0, 1.0, 0.02, 0.02, seed=1, device="cpu")   # A = B = one step: every step
    assert [len(ks.due(k, torch.ones(3, dtype=torch.bool))[0]) for k in range(1, 5)] == [3, 3, 3, 3]


def test_sectors_bins_and_commands():
    from mp4_rl.skills.walker.eval import push as pe
    assert sc.sector_of([0, 22.4, 22.6, 90, 180, 337.4, 337.6, 359.9, -10]).tolist() == [0, 0, 1, 2, 4, 7, 0, 0, 0]
    assert [pe.dir_label(360 * i / 8) for i in range(8)][:3] == ["from_back", "from_back_right", "from_right"]
    assert sc.mag_bins(0.5, 1.5) == [(0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5)]
    assert sc.mag_bins(0.6, 1.1) == [(0.5, 0.75), (0.75, 1.0), (1.0, 1.25)]
    C, names = sc.robot_commands(["stand", "fwd", "left", "right"], 8)
    assert names == ["stand"] * 2 + ["fwd"] * 2 + ["left"] * 2 + ["right"] * 2
    assert C[2].tolist() == pytest.approx([0.1, 0, 0]) and C[5].tolist() == pytest.approx([0, 0.1, 0])


def _synthetic(T=1000, n=5, hold=25):
    """Five robots (see test_kick_accounting) as mode / ok arrays, kicks and entries."""
    mode = np.zeros((T, n), int)
    ok = np.zeros((T - hold + 1, n), bool)
    kicks = {"k": [], "robot": [], "mag": [], "deg": [], "mode": []}
    entries = []
    end_k, end_code = np.full(n, -1), np.zeros(n, int)

    def kick(e, k, mag=1.0, deg=90.0):
        kicks["k"].append(k); kicks["robot"].append(e); kicks["mag"].append(mag); kicks["deg"].append(deg)
        kicks["mode"].append(int(mode[k, e]))

    def getup(e, a, b):
        mode[a:b, e] = 1
        entries.append((a, e))
    # robot 0: kick 100 -> entry 130, GETUP 130..229, walking from 260; a kick at 180 in GETUP; a kick at 400 walking
    ok[:100, 0] = True
    getup(0, 130, 230)
    ok[260:, 0] = True
    # robot 1: kick 100 -> entry 120, GETUP until the timeout at 620
    ok[:100, 1] = True
    getup(1, 120, T)
    end_k[1], end_code[1] = 620, 1
    # robot 2: kick 900 -> entry 920, never walking again, the episode ends 1.5 s after the window's last start
    ok[:900, 2] = True
    getup(2, 920, T)
    # robot 3: kick 50 -> entry 70, GETUP 70..169, walking 200..289; kick 120 in GETUP; kick 190 recovering -> entry
    # 300 (attributed to it), GETUP 300..399, walking from 450
    getup(3, 70, 170)
    ok[200:290, 3] = True
    getup(3, 300, 400)
    ok[450:, 3] = True
    # robot 4: no kicks, a fall at 300 (unattributed), walking from 500
    ok[:300, 4] = True
    getup(4, 300, 420)
    ok[500:, 4] = True
    for e, k in ((0, 100), (0, 180), (0, 400), (1, 100), (2, 900), (3, 50), (3, 120), (3, 190)):
        kick(e, k)
    kicks = {k: np.array(v) for k, v in kicks.items()}
    return mode, ok, kicks, sorted(entries), end_k, end_code


def test_kick_accounting():
    T, hold = 1000, 25
    mode, ok, kicks, entries, end_k, end_code = _synthetic(T)
    acc = sc.kick_accounting(mode, ok, kicks, entries, end_k, end_code, T, hold)
    K, R = acc["kicks"], acc["robots"]
    j = {(int(e), int(k)): i for i, (e, k) in enumerate(zip(kicks["robot"], kicks["k"]))}
    a = j[(0, 100)]
    assert K["state"][a] == "walk" and K["toppled"][a] and K["resolved"][a]
    assert (K["t_detect_s"][a], K["t_getup_s"][a], K["t_handback_s"][a], K["t_walk_s"][a]) == (0.6, 2.0, 0.6, 3.2)
    assert K["within_5s"][a] is True and K["within_8s"][a] is True and K["kicks_while_down"][a] == 1
    assert K["getups"][a] == 1
    b = j[(0, 180)]
    assert K["state"][b] == "getup" and not K["toppled"][b]
    c = j[(0, 400)]
    assert K["state"][c] == "walk" and not K["toppled"][c] and K["t_walk_s"][c] == 0.0
    d = j[(1, 100)]
    assert K["toppled"][d] and not K["resolved"][d] and K["ended"][d]
    assert K["within_8s"][d] is False and K["down_so_far_s"][d] == 10.4
    e = j[(2, 900)]
    assert K["toppled"][e] and K["within_5s"][e] is None and K["within_8s"][e] is None
    f, g, h = j[(3, 50)], j[(3, 120)], j[(3, 190)]
    assert K["toppled"][f] and K["t_walk_s"][f] == 3.0 and K["kicks_while_down"][f] == 2      # 120 and 190
    assert K["state"][g] == "getup" and not K["toppled"][g]
    assert K["state"][h] == "recovering" and K["toppled"][h] and K["t_detect_s"][h] == 2.2
    assert K["t_walk_s"][h] == 5.2 and K["within_5s"][h] is False and K["within_8s"][h] is True
    assert acc["unattributed"] == [[4, 300]]
    assert R["longest_down_s"] == [2.6, 10.0, 1.6, 3.0, 4.0]          # robot 1: until it ended (disarmed)
    assert R["longest_down_open"] == [False, True, True, False, False]
    assert R["end"] == ["", "timeout", "", "", ""] and R["toppled"] == [1, 1, 1, 2, 0]
    assert R["unattributed_falls"] == [0, 0, 0, 0, 1] and R["down_intervals"] == [1, 1, 1, 2, 1]


def test_schedule_metrics_bins():
    """schedule_metrics from per-step arrays (walking = tilt 0 at the command, down = tilt 90): the same accounting,
    the kicks in their magnitude bins and sectors, the summary's numbers."""
    T, n = 1000, 5
    mode, ok, kicks, entries, end_k, end_code = _synthetic(T)
    up = np.zeros((T, n), bool)
    for e in range(n):                                      # a step is "up" if a window covering it qualifies
        for s in np.nonzero(ok[:, e])[0]:
            up[s:s + 25, e] = True
    C = np.zeros((n, 3), np.float32)
    C[:, 0] = 0.1
    kicks["mag"] = np.array([0.55, 0.8, 1.1, 1.3, 1.5, 0.6, 0.9, 1.2])
    kicks["deg"] = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    alive = np.ones((T, n), bool)
    alive[620:, 1] = False
    r = {"mode": mode, "alive": alive, "tilt": np.where(up, 0.0, 90.0), "vx": np.where(up, 0.1, 0.0),
         "vy": np.zeros((T, n)), "C": C, "events": [{"k": k, "robot": e, "event": "getup"} for k, e in entries],
         "kicks": kicks, "end_k": end_k, "end_code": end_code, "currents": None}
    m = sc.schedule_metrics(r, ["fwd"] * n, 0.5, 1.5)
    assert list(m["by_mag"]) == ["0.50-0.75", "0.75-1.00", "1.00-1.25", "1.25-1.50"]
    bm = m["by_mag"]
    assert (bm["0.50-0.75"]["kicks"], bm["0.50-0.75"]["toppled"], bm["0.50-0.75"]["within_8s"]) == (2, 2, 1.0)
    b = bm["1.25-1.50"]                                   # robot 1 (1.3, ended) and robot 2 (1.5, censored)
    assert (b["kicks"], b["toppled"], b["within_8s"], b["evaluable_8s"], b["not_evaluable_8s"], b["ended"]) == \
        (2, 2, 0.0, 1, 1, 1)
    assert bm["0.75-1.00"]["kicks_in_getup"] == 2 and bm["0.75-1.00"]["toppled"] == 0
    assert bm["1.00-1.25"]["toppled"] == 1 and bm["1.00-1.25"]["within_8s"] == 1.0
    assert bm["1.00-1.25"]["t_walk_s_p50"] == 5.2 and bm["1.00-1.25"]["not_toppled_t_walk_s_p50"] == 0.0
    assert m["by_dir"]["from_back"]["kicks"] == 1 and m["by_dir"]["from_right"]["kicks_in_getup"] == 0
    s = m["summary"]
    assert s["robots_down_over_15s"] == 1 and s["robots_ended"] == 1 and s["unattributed_falls"] == 1
    assert s["overall"]["kicks_in_getup"] == 2 and s["overall"]["kicks_recovering"] == 1
    assert s["overall"]["toppled"] == 5 and m["kicks"]["sector"][0] == "from_back"


def test_schedule_in_simulation(tmp_path):
    """Four robots (stand, fwd), kicks 1.5 m/s every 1-1.5 s for 6 s with a stand-seeking get-up: the kicks follow
    the schedule (one generator), every toppling kick has its GETUP entry after it, the JSON has the summary's fields."""
    g = zero_getup(tmp_path / "getup")
    out = tmp_path / "sequence.json"
    rc = sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "4", "--seconds", "6",
                  "--schedule", "1.5", "1.5", "1.0", "1.5", "--sched-commands", "stand,fwd", "--getup-timeout", "3",
                  "--seed", "11", "--json", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["mode"] == "schedule" and d["getup_max"] is None and d["robot_commands"] == ["stand"] * 2 + ["fwd"] * 2
    K = d["kicks"]
    assert len(K["k"]) >= 4 * 3 and set(K["mag_mps"]) == {1.5}
    for i in range(len(K["k"])):
        if K["toppled"][i]:
            assert K["entry_k"][i] > K["k"][i] and K["state"][i] != "getup"
    for rr in range(4):
        kk = [k for k, x in zip(K["k"], K["robot"]) if x == rr]
        assert all(50 <= b - a <= 75 for a, b in zip(kk, kk[1:])) or d["robots"]["end"][rr]
    assert list(d["by_mag"]) == ["1.50-1.75"] or list(d["by_mag"]) == ["1.25-1.50"]
    assert set(d["by_dir"]) == {"from_back", "from_back_right", "from_right", "from_front_right", "from_front",
                                "from_front_left", "from_left", "from_back_left"}
    assert d["currents"]["walk"]["robot_time_s"] > 0
    assert d["summary"]["overall"]["kicks"] == len(K["k"])


CROSS = json.loads((REPO / "tests" / "data" / "phase_restart_cross.json").read_text())["cases"]


@pytest.mark.parametrize("name", sorted(CROSS))
def test_phase_restart_steps_against_the_runners(tmp_path, monkeypatch, name):
    """--getup-phase-restart: the real loop with only the filter's tilt scripted, from the GETUP entry, against
    tests/data/phase_restart_cross.json, which holds the runner's real loop to (no
    one environment has both). Where the two are recorded as different, the case says why."""
    case = CROSS[name]
    entry = 50                                                          # --force-getup-t 1.0

    class ScriptedFilter(sc.TorchGravityFilter):
        calls = 0

        def update(self, gyro, acc, dt):
            ScriptedFilter.calls += 1                                   # call n (from 1) is what step n decides on
            step = ScriptedFilter.calls - entry
            v = 5.0                                                     # walking, upright, before the forced entry
            for frm, tilt in case["tilts"]:
                if step >= frm:
                    v = tilt
            return torch.full((acc.shape[0],), v, dtype=acc.dtype, device=acc.device)

    monkeypatch.setattr(sc, "TorchGravityFilter", ScriptedFilter)
    g = zero_getup(tmp_path / "getup", getup_phase_s=case["phase_s"])
    out = tmp_path / "sequence.json"
    rc = sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "2", "--seconds", str(1.0 + case["steps"] * 0.02),
                  "--kick", "0", "--force-getup-t", "1.0", "--getup-timeout", "60",
                  "--getup-phase-restart", str(case["restart_s"]), "--seed", "11", "--json", str(out)])
    assert rc == 0
    ev = json.loads(out.read_text())["events"]
    assert [e["k"] for e in ev if e["event"] == "getup"] == [entry] * 2
    assert not [e for e in ev if e["event"] == "hand-back"]
    for r in range(2):
        got = [e["k"] - entry for e in ev if e["event"] == "phase-restart" and e["robot"] == r]
        assert got == case["restarts"]["sim"], (r, got)
    want_same = "differs_because" not in case
    assert (case["restarts"]["sim"] == case["restarts"]["runner"]) is want_same


def test_handback_options_reach_the_rules(tmp_path):
    """A tilt threshold no robot can meet keeps them in GETUP; the JSON records the rules in force."""
    g = zero_getup(tmp_path / "getup")
    out = tmp_path / "sequence.json"
    rc = sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "2", "--seconds", "3", "--kick", "0",
                  "--force-getup-t", "1.0", "--handback-tilt", "0", "--handback-gyro", "5", "--handback-rms", "20",
                  "--seed", "11", "--json", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert not [e for e in d["events"] if e["event"] == "hand-back"]
    assert d["rules"]["handback"] == [0.0, 5.0, 20.0, rn.HANDBACK_STEPS]


def test_defaults_are_the_runners(tmp_path):
    g = zero_getup(tmp_path / "getup")
    out = tmp_path / "sequence.json"
    assert sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "1", "--seconds", "1", "--kick", "0",
                    "--seed", "11", "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["getup_timeout_s"] == 20.0 and d["rules"]["handback"] == [rn.HANDBACK_TILT_DEG, rn.HANDBACK_GYRO,
                                                                       rn.HANDBACK_RMS_DEG, rn.HANDBACK_STEPS]


def test_reach_in_simulation(tmp_path, capsys):
    """Two robots forced into GETUP report every leg against the proven reach, in the JSON and printed."""
    g = zero_getup(tmp_path / "getup")
    out = tmp_path / "sequence.json"
    rc = sc.main(["--walker", str(WALKER), "--getup", str(g), "--robots", "2", "--seconds", "2", "--kick", "0",
                  "--force-getup-t", "1.0", "--seed", "11", "--reach", "--json", str(out)])
    assert rc == 0
    reach = json.loads(out.read_text())["reach"]
    assert set(reach["legs"]) == {"FR", "FL", "RR", "RL"}
    for r in reach["legs"].values():                # a stand-seeking get-up stays near the stand
        assert abs(r["commanded_max_deg"]) < 20.0 and abs(r["measured_max_deg"]) < 20.0
    assert "reach RR: commanded" in capsys.readouterr().out


def fall_run(T=400, n=6):
    """Tilt and events of six falls: the rows ``test_route_*`` read."""
    tilt = np.full((T, n), 5.0)
    ev = []

    def getup(r, k, gy, peak, up_at=None, restart_at=None):
        ev.append({"k": k, "robot": r, "event": "getup", "gravity": [0.0, gy, -0.7]})
        end = up_at or restart_at or T
        tilt[k:end, r] = np.linspace(50.0, peak, end - k)
        if up_at:
            ev.append({"k": up_at, "robot": r, "event": "hand-back"})
        if restart_at:
            ev.append({"k": restart_at, "robot": r, "event": "phase-restart"})
            tilt[restart_at:, r] = 178.0                 # what happens after the first attempt doesn't count
    getup(0, 10, -0.7, 176.0, up_at=110)                 # right, via the back, up in 2 s
    getup(1, 10, -0.7, 90.0, up_at=60)                   # right, direct, up in 1 s
    getup(2, 10, 0.7, 150.0, restart_at=260)             # left, between the two, not up in its attempt
    getup(3, 10, 0.05, 176.0, up_at=210)                 # neither flank, via the back
    getup(4, T - 100, 0.7, 176.0)                        # too near the end: left out
    return ev, tilt


def test_route_on_a_fabricated_run():
    ev, tilt = fall_run()
    s = sc.route_summary(ev, tilt)
    a, right, left = s["all"], s["right_lower"], s["left_lower"]
    assert (a["n"], right["n"], left["n"]) == (4, 2, 1)
    assert a["over_120"] == 0.75 and a["over_170"] == 0.5 and a["up_in_attempt"] == 0.75
    assert right["over_120"] == 0.5 and right["over_170"] == 0.5 and right["t_via_back_p50_s"] == 2.0
    assert right["t_direct_p50_s"] == 1.0 and left["t_direct_p50_s"] is None and left["t_via_back_p50_s"] is None
    assert left["over_120"] == 1.0 and left["over_170"] == 0.0 and left["up_in_attempt"] == 0.0
    assert a["t_p50_s"] == 3.0 and a["t_via_back_p50_s"] == 3.0
    assert a["over_170_in_1s"] == 0.0                     # reached later than its first second


def test_route_ends_an_attempt_when_its_robot_ends():
    tilt = np.full((400, 1), 5.0)
    tilt[50:100, 0] = 90.0
    tilt[100:, 0] = 178.0
    ev = [{"k": 50, "robot": 0, "event": "getup", "gravity": [0.0, -0.7, -0.7]}]
    assert sc.route_summary(ev, tilt)["all"]["over_170"] == 1.0
    s = sc.route_summary(ev, tilt, np.array([100]))["all"]
    assert s["over_170"] == 0.0 and s["t_p50_s"] == 1.0


def test_route_of_no_falls():
    assert sc.route_summary([], np.zeros((100, 2)))["all"] == {
        "n": 0, "over_120": None, "over_170": None, "over_170_in_1s": None, "up_in_attempt": None, "t_p50_s": None,
        "t_direct_p50_s": None, "t_via_back_p50_s": None}


def test_never_up_and_slow_up():
    """Limp, still down after over 15 s, up after over 15 s; a robot that rose late and is down again for good is never
    up; together they are every robot not up at the end."""
    T, hold = 3000, 25
    n = 6
    mode = np.zeros((T, n), int)
    ok = np.zeros((T - hold + 1, n), bool)
    ok[:, :] = True
    entries, end_k, end_code = [], np.full(n, -1), np.zeros(n, int)

    def down(e, a, b):
        mode[a:b, e] = 1
        ok[a:b, e] = False
        entries.append((a, e))
    down(0, 100, 200)                                    # 2 s
    down(1, 100, T)                                      # still down, 58 s
    down(2, 100, 1101)                                   # rose after 20 s
    down(3, 100, 1100)                                   # limp at 20 s
    end_k[3], end_code[3] = 1100, 1
    ok[1100:, 3] = False
    down(4, 100, 1000)                                   # rose after 18 s, then down for the rest: 20 s
    down(4, 2000, T)
    down(5, T - 200, T)                                  # still down, only 4 s
    kicks = {k: np.zeros(0, int) for k in ("k", "robot", "mag", "deg", "mode")}
    acc = sc.kick_accounting(mode, ok, kicks, sorted(entries), end_k, end_code, T, hold)
    R = acc["robots"]
    assert R["open_down_s"] == [0.0, 58.0, 0.0, 20.0, 20.0, 4.0]
    assert R["longest_closed_down_s"] == [2.0, 0.0, 20.02, 0.0, 18.0, 0.0]
    assert sc.down_counts(R) == {"never_up": 3, "slow_up": 1}
