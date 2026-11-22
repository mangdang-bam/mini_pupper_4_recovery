"""Gait-quality metrics from a rollout's per-step records (50 Hz): support for judging gait character on video, not a
substitute for it.

Inputs per env (``GaitRecorder`` collects them in ``omni_eval``): the feet's contact flags (FR, FL, RR, RL; the
``feet_contact`` sensor at the end of each policy step), the foot sites' world positions and heights above the terrain
under them, the joint targets in force (the servo term's model targets, rad), and which steps are measured (after the
settle, before the env's first end). Leg names and order: FR, FL, RR, RL.

Definitions (per env and leg, over its measured window; contacts are sampled every 20 ms, so a 360 ms trot cycle is 18
samples and one sample is 0.056 of it):

- contacts are debounced first (``debounce``): a one-sample contact inside a swing (a scuff, 20 ms) is dropped, then a
  one-sample gap inside a contact (a foot bouncing) is filled.
- touchdown: a step whose contact is on after a step with it off; lift-off: the reverse. A "cycle" runs from one
  touchdown to the next.
- ``period_s``: the mean time between consecutive touchdowns of a leg, averaged over legs. ``period_cv``: per leg the
  coefficient of variation (std / mean) of those intervals, averaged over legs (cycle-to-cycle variability; 0 =
  metronomic).
- ``duty``: per leg the share of samples in contact between its first and last touchdown (whole cycles only), averaged
  over legs; ``duty_per_leg``.
- phases: each touchdown of leg B is placed in leg A's cycle that contains it, as (t_B - t_A) / (A's next touchdown -
  t_A), in [0, 1); the circular mean over touchdowns. ``phase_diag_within``: B vs A for the two diagonal pairs (RL vs
  FR, RR vs FL), each wrapped to [-0.5, 0.5) and the absolute values averaged (a trot: 0). ``phase_diag_pairs``: the
  pair (FL, RR) against the pair (FR, RL): the circular mean of FL vs FR and RR vs RL (a trot: 0.5).
  ``phase_left_right``: FL vs FR and RL vs RR, circular mean (a trot or a pace: 0.5; a bound: 0).
- ``stride_m``: per leg the mean horizontal distance (world) between consecutive touchdown positions;
  ``stride_sym``: |left - right| / mean(left, right), left = mean of FL and RL, right = FR and RR (0 = symmetric).
  ``duty_lr_diff``: |duty left - duty right|.
- ``clearance_mm``: per swing (a lift-off to the next touchdown) the highest foot site height above the terrain under it
  less the site's height at contact on flat ground (``foot_r``), averaged over swings and legs.
- ``jerk_rad``: the rms over joints and steps of the second difference of the joint targets sampled at 50 Hz (rad per
  step^2; x[t+1] - 2 x[t] + x[t-1]); ``jerk_deg`` the same in degrees.

An env contributes when every leg has at least three touchdowns in its window (``n_gait``); the means are over those
envs. A leg that never lifts (a stand, a shuffle without contact changes) leaves its env out.
"""
from __future__ import annotations

import math

import numpy as np

LEGS = ("FR", "FL", "RR", "RL")
DT = 0.02
MIN_TOUCHDOWNS = 3


def debounce(c) -> np.ndarray:
    """(T,) or (T, k) contact: one-sample contacts removed, then one-sample gaps filled."""
    c = np.asarray(c, dtype=bool).copy()
    if len(c) < 3:
        return c
    blip = c[1:-1] & ~c[:-2] & ~c[2:]
    c[1:-1] &= ~blip
    gap = ~c[1:-1] & c[:-2] & c[2:]
    c[1:-1] |= gap
    return c


def touchdowns(c) -> np.ndarray:
    c = np.asarray(c, dtype=bool)
    return np.nonzero(c[1:] & ~c[:-1])[0] + 1


def liftoffs(c) -> np.ndarray:
    c = np.asarray(c, dtype=bool)
    return np.nonzero(~c[1:] & c[:-1])[0] + 1


def circ_mean(ph) -> float:
    """Circular mean of phases in [0, 1) (nan when empty)."""
    ph = np.asarray(ph, dtype=float)
    if ph.size == 0:
        return float("nan")
    a = 2 * np.pi * ph
    return float((math.atan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi)) % 1.0)


def relative_phases(td_a, td_b) -> np.ndarray:
    """Each touchdown of B inside a cycle of A (between two of A's touchdowns): its phase in A's cycle."""
    td_a, td_b = np.asarray(td_a), np.asarray(td_b)
    out = []
    for t in td_b:
        k = np.searchsorted(td_a, t, side="right") - 1
        if k < 0 or k + 1 >= len(td_a):
            continue
        out.append((t - td_a[k]) / (td_a[k + 1] - td_a[k]))
    return np.asarray(out, dtype=float) % 1.0


def wrap_half(x: float) -> float:
    return (x + 0.5) % 1.0 - 0.5


def env_metrics(contact, foot_xy=None, foot_h=None, targets=None, foot_r: float = 0.0, dt: float = DT) -> dict | None:
    """One env's metrics over its measured window. ``contact`` (T, 4) bool; ``foot_xy`` (T, 4, 2) m (world);
    ``foot_h`` (T, 4) m (site above the terrain); ``targets`` (T, 12) rad. None when a leg has fewer than
    ``MIN_TOUCHDOWNS`` touchdowns."""
    contact = debounce(contact)
    td = [touchdowns(contact[:, j]) for j in range(4)]
    if min(len(t) for t in td) < MIN_TOUCHDOWNS:
        return None
    periods, cvs, duty, stride, clear = [], [], [], [], []
    for j in range(4):
        iv = np.diff(td[j]) * dt
        periods.append(iv.mean())
        cvs.append(iv.std() / iv.mean())
        duty.append(contact[td[j][0]:td[j][-1], j].mean())
        if foot_xy is not None:
            p = np.asarray(foot_xy)[td[j], j]
            stride.append(float(np.linalg.norm(np.diff(p, axis=0), axis=-1).mean()))
        if foot_h is not None:
            lo = liftoffs(contact[:, j])
            peaks = []
            for a in lo:
                nxt = td[j][td[j] > a]
                if len(nxt):
                    peaks.append(np.asarray(foot_h)[a:nxt[0] + 1, j].max() - foot_r)
            if peaks:
                clear.append(float(np.mean(peaks)))
    ph = lambda a, b: relative_phases(td[LEGS.index(a)], td[LEGS.index(b)])   # noqa: E731  B in A's cycle
    within = [abs(wrap_half(circ_mean(ph("FR", "RL")))), abs(wrap_half(circ_mean(ph("FL", "RR"))))]
    pairs = circ_mean(np.concatenate([ph("FR", "FL"), ph("RL", "RR")]))
    lr = circ_mean(np.concatenate([ph("FR", "FL"), ph("RR", "RL")]))
    dl, dr = 0.5 * (duty[1] + duty[3]), 0.5 * (duty[0] + duty[2])
    r = {"period_s": float(np.mean(periods)), "period_cv": float(np.mean(cvs)), "duty": float(np.mean(duty)),
         "duty_per_leg": [float(d) for d in duty], "duty_lr_diff": float(abs(dl - dr)),
         "phase_diag_within": float(np.nanmean(within)), "phase_diag_pairs": pairs, "phase_left_right": lr}
    if stride:
        sl, sr = 0.5 * (stride[1] + stride[3]), 0.5 * (stride[0] + stride[2])
        r.update(stride_m=float(np.mean(stride)), stride_per_leg_m=stride,
                 stride_sym=float(abs(sl - sr) / max(0.5 * (sl + sr), 1e-9)))
    if clear:
        r["clearance_mm"] = 1000.0 * float(np.mean(clear))
    if targets is not None:
        x = np.asarray(targets, dtype=float)
        if len(x) >= 3:
            j2 = x[2:] - 2 * x[1:-1] + x[:-2]
            r["jerk_rad"] = float(np.sqrt(np.mean(j2 ** 2)))
            r["jerk_deg"] = float(np.degrees(r["jerk_rad"]))
    return r


def aggregate(rows) -> dict:
    """Means over envs (circular means for the phases); ``n_gait`` the envs that contributed."""
    rows = [r for r in rows if r is not None]
    out = {"n_gait": len(rows)}
    if not rows:
        return out
    for k in rows[0]:
        vals = [r[k] for r in rows if k in r]
        if k.endswith("_per_leg") or k.endswith("_per_leg_m"):
            out[k] = [float(v) for v in np.mean(np.asarray(vals, dtype=float), axis=0)]
        elif k in ("phase_diag_pairs", "phase_left_right"):
            out[k] = circ_mean(vals)
        else:
            out[k] = float(np.mean(vals))
    return out


def window_metrics(contact, live, foot_xy=None, foot_h=None, targets=None, foot_r: float = 0.0, k0: int = 0,
                   ids=None) -> dict:
    """Metrics for envs ``ids`` from full-run arrays (T first): each env from step ``k0`` to its last live step."""
    contact = np.asarray(contact)
    n = contact.shape[1]
    ids = range(n) if ids is None else ids
    rows = []
    for e in ids:
        lv = np.asarray(live)[k0:, e]
        end = k0 + (int(np.argmin(lv)) if not lv.all() else len(lv))
        sl = slice(k0, end)
        rows.append(env_metrics(contact[sl, e], None if foot_xy is None else np.asarray(foot_xy)[sl, e],
                                None if foot_h is None else np.asarray(foot_h)[sl, e],
                                None if targets is None else np.asarray(targets)[sl, e], foot_r))
    return aggregate(rows)


class GaitRecorder:
    """Collects the per-step records on the GPU (``__call__(env)`` after every ``env.step``) and returns numpy arrays
    (``arrays()``): contact (T, n, 4), foot_xy (T, n, 4, 2), foot_h (T, n, 4), targets (T, n, 12)."""

    def __init__(self):
        self.c, self.xy, self.h, self.tg = [], [], [], []

    def __call__(self, env) -> None:
        from mp4_rl.common.env import _foot_site_ids, foot_heights_centre
        from mp4_rl.common.mdp import servo_action
        robot = env.scene["robot"]
        self.c.append((env.scene["feet_contact"].data.found > 0).reshape(env.num_envs, -1)[:, :4].clone())
        self.xy.append(robot.data.site_pos_w[:, _foot_site_ids(env), :2].clone())
        self.h.append(foot_heights_centre(env).clone())
        self.tg.append(servo_action(env).target_model.clone())

    def arrays(self) -> dict:
        import torch
        st = lambda L: torch.stack(L).cpu().numpy()   # noqa: E731
        return {"contact": st(self.c), "foot_xy": st(self.xy), "foot_h": st(self.h), "targets": st(self.tg)}
