"""Gait-quality metrics (``eval/gait_metrics.py``) on synthetic contact sequences with known answers."""
import numpy as np
import pytest

from mp4_rl.walker.eval import gait_metrics as gm


def pattern(T, period, duty, offsets):
    """(T, 4) contact: each leg on for ``duty`` of a ``period``-step cycle, its touchdown at ``offsets[leg]`` steps."""
    t = np.arange(T)[:, None]
    ph = ((t - np.asarray(offsets)[None, :]) % period) / period
    return ph < duty


def test_ideal_trot():
    c = pattern(180, 18, 0.75, [0, 9, 9, 0])          # FR, FL, RR, RL: the diagonal pairs half a cycle apart
    r = gm.env_metrics(c)
    assert r["period_s"] == pytest.approx(0.36)
    assert r["period_cv"] == pytest.approx(0.0)
    assert r["duty"] == pytest.approx(14 / 18)            # steps 0..13 of 18 (ph < 0.75)
    assert r["phase_diag_within"] == pytest.approx(0.0, abs=1e-9)
    assert r["phase_diag_pairs"] == pytest.approx(0.5)
    assert r["phase_left_right"] == pytest.approx(0.5)
    assert r["duty_lr_diff"] == pytest.approx(0.0)


def test_pace_and_bound_are_told_apart():
    pace = gm.env_metrics(pattern(180, 18, 0.6, [0, 9, 0, 9]))     # right legs together, left legs together
    assert pace["phase_left_right"] == pytest.approx(0.5)
    assert pace["phase_diag_within"] == pytest.approx(0.5)
    bound = gm.env_metrics(pattern(180, 18, 0.6, [0, 0, 9, 9]))    # fronts together, rears together
    assert bound["phase_left_right"] == pytest.approx(0.0, abs=1e-9)
    assert bound["phase_diag_within"] == pytest.approx(0.5)


def test_variability_and_asymmetry():
    rng = np.random.default_rng(0)
    T = 400
    c = np.zeros((T, 4), dtype=bool)
    t = 0
    starts = []
    while t < T - 30:
        starts.append(t)
        t += int(rng.choice([16, 18, 20]))
    for leg, (off, on) in enumerate([(0, 14), (9, 12), (9, 14), (0, 12)]):   # left legs (FL, RL) 12 of each cycle
        for s in starts:
            c[s + off:s + off + on, leg] = True
    r = gm.env_metrics(c)
    assert r["period_cv"] > 0.03
    dl = 0.5 * (r["duty_per_leg"][1] + r["duty_per_leg"][3])
    dr = 0.5 * (r["duty_per_leg"][0] + r["duty_per_leg"][2])
    assert r["duty_lr_diff"] == pytest.approx(abs(dl - dr))
    assert r["duty_lr_diff"] > 0.05


def test_stride_clearance_jerk():
    T, period = 180, 18
    c = pattern(T, period, 0.75, [0, 9, 9, 0])
    xy = np.zeros((T, 4, 2))
    h = np.full((T, 4), 0.005)
    for leg, step in enumerate([0.04, 0.03, 0.04, 0.03]):            # right legs 40 mm strides, left 30 mm
        td = gm.touchdowns(c[:, leg])
        x = 0.0
        for k in range(T):
            if k in td:
                x += step
            xy[k, leg, 0] = x
        for a in gm.liftoffs(c[:, leg]):
            h[a + 2, leg] = 0.005 + 0.02                              # a 20 mm apex per swing
    tg = 0.5 * 0.01 * np.arange(T)[:, None] ** 2 * np.ones((1, 12))   # second difference 0.01 rad everywhere
    r = gm.env_metrics(c, xy, h, tg, foot_r=0.005)
    assert r["stride_per_leg_m"] == pytest.approx([0.04, 0.03, 0.04, 0.03])
    assert r["stride_sym"] == pytest.approx(0.01 / 0.035)
    assert r["clearance_mm"] == pytest.approx(20.0)
    assert r["jerk_rad"] == pytest.approx(0.01)


def test_standing_robot_is_left_out_and_windows():
    c = np.ones((100, 4), dtype=bool)
    assert gm.env_metrics(c) is None
    trot = pattern(200, 18, 0.75, [0, 9, 9, 0])
    contact = np.stack([trot, np.ones_like(trot)], axis=1)            # (T, 2 envs, 4)
    live = np.ones((200, 2), dtype=bool)
    live[150:, 0] = False                                              # env 0 ends at step 150
    out = gm.window_metrics(contact, live, k0=20)
    assert out["n_gait"] == 1
    assert out["period_s"] == pytest.approx(0.36)


def test_circular_mean_wraps():
    assert gm.circ_mean([0.95, 0.05]) == pytest.approx(0.0, abs=1e-9) or gm.circ_mean([0.95, 0.05]) == pytest.approx(1.0)
    assert gm.aggregate([{"phase_left_right": 0.95, "duty": 0.7}, {"phase_left_right": 0.05, "duty": 0.8}])[
        "duty"] == pytest.approx(0.75)


def test_debounce_bounce_and_scuff():
    c = pattern(180, 18, 0.75, [0, 9, 9, 0])
    noisy = c.copy()
    noisy[5, 0] = False                       # a bounce inside FR's contact
    noisy[16, 0] = True                       # a scuff inside FR's swing (steps 14..17 off)
    assert gm.env_metrics(noisy)["period_s"] == pytest.approx(0.36)
    assert gm.env_metrics(noisy)["period_cv"] == pytest.approx(0.0)
