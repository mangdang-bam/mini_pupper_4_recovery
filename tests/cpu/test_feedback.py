"""The position feedback's plausibility filter (deploy/feedback.py), alone, on robot logs, and where the link uses it."""
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from mp4_recovery.deploy import feedback as fb

DATA = Path(__file__).parent / "data"


def ramp(n=200, start=1500.0, per_tick=3.0):
    return np.stack([start + per_tick * np.arange(n) + 10 * j for j in range(12)], axis=1)


def test_single_frame_glitches_are_held_and_counted():
    clean = ramp()
    raw = clean.copy()
    raw[50, 1] = 0                       # a zero
    raw[80, 2] = 2 * clean[80, 2]        # doubled: past 3100
    raw[81, 2] = 47104                   # the next frame too, another value
    raw[120, [6, 7, 8]] = [0, 3308, 6222]
    raw[150, 4] = clean[150, 4] - 1000   # in range, an impossible step
    out, rej, f = fb.filter_log(raw)
    want = np.zeros(raw.shape, bool)
    want[50, 1] = want[80, 2] = want[81, 2] = want[150, 4] = True
    want[120, [6, 7, 8]] = True
    assert np.array_equal(rej, want)
    assert out[50, 1] == clean[49, 1] and out[80, 2] == out[81, 2] == clean[79, 2]   # the last accepted value
    assert np.array_equal(out[120, [6, 7, 8]], clean[119, [6, 7, 8]]) and out[150, 4] == clean[149, 4]
    assert np.array_equal(out[~rej], clean[~rej])
    assert f.rejected.sum() == 7 and f.frames_rejected == 5 and f.relocked.sum() == 0
    assert "7 readings rejected in 5 of 200 STATE frames" in f.report()


def test_a_real_jump_is_accepted_on_the_third_consistent_frame():
    f = fb.FeedbackFilter(n=1)
    seq = [1500, 1502, 2500, 2501, 2503, 2504]
    res = [f.update([v], tick=k) for k, v in enumerate(seq)]
    assert [bool(r[1][0]) for r in res] == [False, False, True, True, False, False]
    assert [float(r[0][0]) for r in res] == [1500, 1502, 1502, 1502, 2503, 2504]
    assert f.relocked.tolist() == [1]


def test_inconsistent_readings_never_relock():
    f = fb.FeedbackFilter(n=1)
    f.update([1500], tick=0)
    for k, v in enumerate([0, 3308, 600, 2900, 47104, 10], start=1):
        out, rej = f.update([v], tick=k)
        assert rej[0] and out[0] == 1500
    assert f.relocked.tolist() == [0]


def test_a_persistent_reading_past_the_end_is_accepted_as_read():
    f = fb.FeedbackFilter(n=1)
    f.update([3050], tick=0)
    res = [f.update([3112], tick=k) for k in range(1, 6)]
    assert [bool(r[1][0]) for r in res] == [True, True, False, False, False]
    assert [float(r[0][0]) for r in res] == [3050, 3050, 3112, 3112, 3112]
    assert f.relocked.tolist() == [1]


def test_the_step_scales_with_the_ticks_elapsed_up_to_the_cap():
    f = fb.FeedbackFilter(n=1)
    f.update([1000], tick=10)
    assert not f.update([1590], tick=13)[1][0]                # 3 ticks: up to 600 dd
    assert f.update([2400], tick=23)[1][0]                    # 10 ticks, capped at 4: up to 800 dd
    g = fb.FeedbackFilter(n=1)
    g.update([1000], tick=0xFFFFFFFF)
    assert not g.update([1150], tick=0)[1][0]                 # the tick counter wraps


def test_joints_of_a_stale_board_are_not_judged():
    f = fb.FeedbackFilter()
    f.update(np.full(12, 1500.0), tick=0)
    x = np.full(12, 1500.0)
    x[0:3] = 0                                                 # board 0 (FR) not fresh: a boot's zeros
    out, rej = f.update(x, tick=1, fresh=fb.fresh_joints(0b1110))
    assert not rej.any() and np.all(out == 1500.0) and f.rejected.sum() == 0
    assert fb.fresh_joints(0x0F).all() and fb.fresh_joints(0b0100).tolist() == [False] * 6 + [True] * 3 + [False] * 3


def test_position_falls_back_to_the_raw_reading():
    class S:
        pos_dd = np.array([1, 2], dtype=np.uint16)
    s = S()
    assert fb.position(s).tolist() == [1.0, 2.0]
    s.pos_filt = np.array([3.0, 4.0])
    assert fb.position(s).tolist() == [3.0, 4.0]


def test_robot_log_excerpts():
    """A limp hand-moved stretch with ten frames of impossible readings, and a still stretch with none."""
    d = np.load(DATA / "feedback_glitches.npz")
    out, rej, f = fb.filter_log(d["map_pos_dd"], d["map_tick"], d["map_boards"])
    p = d["map_pos_dd"].astype(float)
    glitch = np.zeros(p.shape, bool)
    glitch[1:-1] = (np.abs(p[1:-1] - p[:-2]) > 200) & (np.abs(p[1:-1] - p[2:]) > 200)
    glitch |= p > 3100
    assert f.frames_rejected == 10 and rej.any(1).sum() == 10
    assert not (rej & ~glitch).any() and not (glitch & ~rej).any()
    assert np.abs(np.diff(out, axis=0)).max() < 200 and f.relocked.sum() == 0
    _, rej2, f2 = fb.filter_log(d["still_pos_dd"], d["still_tick"], d["still_boards"])
    assert not rej2.any() and f2.frames == 3000


LOGS = os.environ.get("MP4_ROBOT_LOGS")      # a folder holding whole robot logs map.npz, still60.npz, calib_a5.npz


@pytest.mark.skipif(not LOGS, reason="MP4_ROBOT_LOGS not set")
@pytest.mark.parametrize("name, frames", [("map.npz", 82), ("still60.npz", 0), ("calib_a5.npz", 0)])
def test_whole_robot_logs(name, frames):
    d = np.load(Path(LOGS) / name)
    _, rej, f = fb.filter_log(d["pos_dd"], d["tick"], d["boards"])
    assert f.frames_rejected == frames


# ---- where the link uses it --------------------------------------------------------------------------------------
from fake_esp32 import FakeEsp32  # noqa: E402
from mp4_recovery.deploy.mp4link import cli  # noqa: E402
from mp4_recovery.deploy.mp4link import protocol as p  # noqa: E402
from robot_helpers import C  # noqa: E402


@pytest.fixture
def fake(fixed_calibration):
    f = FakeEsp32()
    f.offsets = {i + 1: float(v) for i, v in enumerate(C.offsets_deg)}
    f.pos = [float(v) for v in C.stand_dd]
    for s in range(1, 13):
        f.params[s][p.PARAM_ID["kff_current"]] = 0.001
    yield f
    f.close()


def glitch_every(fake, period, servo, value, after=None):
    """A one-frame reading ``value`` on ``servo`` (0-based) every ``period`` frames (once ``after()`` holds)."""
    orig = fake.before_tick
    n = {"k": 0, "keep": None}

    def tick():
        orig()
        if n["keep"] is not None:
            fake.pos[servo] = n["keep"]
            n["keep"] = None
        n["k"] += 1
        if (after is None or after()) and n["k"] % period == 0:
            n["keep"] = fake.pos[servo]
            fake.pos[servo] = value
    fake.before_tick = tick


def test_link_logs_raw_readings_and_the_reject_mask(fake, tmp_path, capsys):
    glitch_every(fake, 40, 4, 0)
    assert cli.main(["--port", fake.port, "ping", "--seconds", "1", "--log", str(tmp_path / "p.npz")]) == 0
    d = np.load(tmp_path / "p.npz")
    meta = json.loads(str(d["meta"]))
    rej = d["pos_reject"]
    assert rej.shape == d["pos_dd"].shape and rej[:, 4].sum() >= 3 and not np.delete(rej, 4, axis=1).any()
    assert np.all(d["pos_dd"][rej[:, 4], 4] == 0)                     # the raw readings kept
    assert meta["feedback_filter"]["rejected_per_servo"][4] == int(rej[:, 4].sum())
    assert "feedback filter: %d readings rejected" % rej.sum() in capsys.readouterr().out


def test_stream_judges_the_track_end_on_filtered_readings(fake, tmp_path, capsys):
    glitch_every(fake, 50, 1, 3090, after=lambda: fake.armed)                       # one frame near the end, every 0.25 s
    f = tmp_path / "s.npz"
    np.savez(f, delta_dd=np.zeros((150, 12)), period_s=0.02)
    log = tmp_path / "run.npz"
    assert cli.main(["--port", fake.port, "stream", str(f), "--log", str(log)]) == 0
    meta = json.loads(str(np.load(log)["meta"]))
    assert meta["stop_reason"] == "completed" and meta["track_end_servo"] is None
    assert np.load(log)["pos_reject"][:, 1].sum() >= 10
    out = capsys.readouterr().out
    assert "feedback filter:" in out and "readings rejected" in out


def test_watch_ignores_a_glitch_in_its_min_max_and_near_end_flag(fake, tmp_path, monkeypatch, capsys):
    r, w = os.pipe()
    monkeypatch.setattr(cli, "_stdin_fd", lambda: r)
    glitch_every(fake, 7, 2, 0)
    log = tmp_path / "w.npz"
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rc", cli.main(
        ["--port", fake.port, "watch", "--seconds", "1.6", "--log", str(log)])))
    t.start()
    t.join(10)
    os.close(r)
    os.close(w)
    assert out["rc"] == 0
    d = np.load(log)
    meta = json.loads(str(d["meta"]))
    assert meta["min_dd"][2] == meta["max_dd"][2] == float(C.stand_dd[2])
    assert d["pos_reject"][:, 2].sum() > 20 and (d["pos_dd"][:, 2] == 0).sum() == d["pos_reject"][:, 2].sum()
    text = capsys.readouterr().out
    assert "near track end" not in text and "feedback filter:" in text


def test_runner_observes_filtered_positions(fixed_calibration, monkeypatch, tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).parent))
    import robot_helpers as rh
    from fake_esp32 import FakeRobot
    from mp4_recovery.deploy import runner as rn
    monkeypatch.setattr(rn, "HOLD_AFTER_S", 0.2)
    robot = FakeRobot(C, rh.MOUNT, rn.IDENTIFIED)
    try:
        glitch_every(robot, 9, 5, 0, after=lambda: len(robot.armed_seen) > 5)
        rc, z, meta = rh.go(robot, tmp_path, seconds=1.5)
    finally:
        robot.close()
    assert rc == 0 and meta["stop_reason"] == "completed"
    obs = np.asarray(z["obs"])
    d = np.load(tmp_path / "run.npz")
    assert d["pos_reject"][:, 5].sum() > 10 and json.loads(str(d["meta"]))["feedback_filter"]["frames_rejected"] > 10
    # the observed position and velocity of the servo: no trace of the glitch's 0 dd (rel ~2.6 rad from the stand)
    assert np.abs(obs[:, 9 + 5]).max() < 0.5 and np.abs(obs[:, 21 + 5]).max() < 5.0
    assert "feedback filter:" in capsys.readouterr().out
