"""A plausibility filter for the servos' position feedback (STATE ``pos_dd``).

While the joints move, a STATE frame now and then carries impossible readings on one to three joints of one servo board
(0, twice or several times the true value, values above 3100 dd), for that one frame. Per joint, a reading is rejected
when it lies outside ``LO_DD``..``HI_DD``, or differs from the last accepted reading by more than ``STEP_DD`` per STATE
tick elapsed since it (5 ms ticks; the ticks counted up to ``CAP_TICKS``). A rejected reading leaves the last accepted
value in place. A reading that stays away, ``CONFIRM`` consecutive frames each within the step of the one before, is
accepted on the last of them and while it stays consistent (a real fast move or a board that comes back is not locked
out); so is one outside the range (a joint really past its track end is reported, not held at its last value).
Joints of a board without fresh feedback in that frame (``boards`` bit clear) are not judged: they keep their state and
are not counted. Before a joint's first accepted reading, its output is the raw reading.

``FeedbackFilter.update`` is the per-frame form (the serial link runs one on every STATE frame); ``filter_log`` replays a
recorded log. Logs keep the raw ``pos_dd``; ``pos_reject`` marks the rejected readings.
"""
from __future__ import annotations

import numpy as np

LO_DD, HI_DD = 0.0, 3100.0     # the potentiometer's readable track
STEP_DD = 200.0                # the most a joint may move per STATE tick (20 deg in 5 ms)
CAP_TICKS = 4                  # the step scales with the ticks elapsed up to this many
CONFIRM = 3                    # consecutive consistent frames away from the held value that are accepted
N_SERVOS = 12
WRAP = 1 << 32


_FRESH = [np.array([(b >> (i // 3)) & 1 for i in range(N_SERVOS)], dtype=bool) for b in range(16)]


def fresh_joints(boards: int | None, n: int = N_SERVOS) -> np.ndarray | None:
    """Per servo: whether its board's bit is set in a STATE frame's ``boards`` (bit b = servos 3b+1..3b+3)."""
    if boards is None:
        return None
    if n == N_SERVOS:
        return _FRESH[int(boards) & 0x0F]
    return np.array([(int(boards) >> (i // 3)) & 1 for i in range(n)], dtype=bool)


class FeedbackFilter:
    def __init__(self, n: int = N_SERVOS, step_dd: float = STEP_DD, cap_ticks: int = CAP_TICKS,
                 confirm: int = CONFIRM, lo_dd: float = LO_DD, hi_dd: float = HI_DD):
        self.n, self.step_dd, self.cap_ticks, self.confirm = int(n), float(step_dd), int(cap_ticks), int(confirm)
        self.lo_dd, self.hi_dd = float(lo_dd), float(hi_dd)
        self.reset()

    def reset(self) -> None:
        n = self.n
        self.value = np.full(n, np.nan)              # last accepted reading
        self.at = np.zeros(n, dtype=np.int64)        # its tick
        self.cand = np.full(n, np.nan)               # the latest reading away from the accepted value
        self.cand_at = np.zeros(n, dtype=np.int64)
        self.cand_n = np.zeros(n, dtype=np.int64)    # consecutive consistent readings away, the latest included
        self._tick = -1                              # the frame counter when no tick is given
        self.frames = 0
        self.frames_rejected = 0
        self.rejected = np.zeros(n, dtype=np.int64)
        self.relocked = np.zeros(n, dtype=np.int64)  # moves accepted after persisting away (once per move)

    def _elapsed(self, tick: int, since: np.ndarray) -> np.ndarray:
        d = (tick - since) % WRAP
        d = np.where(d >= WRAP // 2, 1, d)           # an older tick: as one
        return np.clip(d, 1, self.cap_ticks)

    def update(self, pos_dd, tick: int | None = None, fresh=None) -> tuple[np.ndarray, np.ndarray]:
        """One frame: ``pos_dd`` (n readings), its STATE ``tick`` (None: one tick after the previous frame), ``fresh``
        (n bools, None: all). Returns (the filtered positions, float dd; the rejected readings, bool)."""
        x = np.asarray(pos_dd, dtype=float).reshape(self.n)
        tick = self._tick + 1 if tick is None else int(tick)
        self._tick = tick
        judge = np.ones(self.n, dtype=bool) if fresh is None else np.asarray(fresh, dtype=bool).reshape(self.n)
        inr = (x >= self.lo_dd) & (x <= self.hi_dd)
        have = np.isfinite(self.value)
        near = have & (np.abs(x - self.value) <= self.step_dd * self._elapsed(tick, self.at))
        accept = judge & inr & (~have | near)
        away = judge & ~accept
        consistent = (self.cand_n > 0) & (np.abs(x - self.cand) <= self.step_dd * self._elapsed(tick, self.cand_at))
        cand_n = np.where(away, np.where(consistent, self.cand_n + 1, 1), np.where(judge, 0, self.cand_n))
        relock = away & (cand_n >= self.confirm)
        accept |= relock
        self.cand_n = cand_n
        self.cand = np.where(away, x, self.cand)
        self.cand_at = np.where(away, tick, self.cand_at)
        self.value = np.where(accept, x, self.value)
        self.at = np.where(accept, tick, self.at)
        reject = judge & ~accept
        self.frames += 1
        self.frames_rejected += int(reject.any())
        self.rejected += reject
        self.relocked += relock & (cand_n == self.confirm)
        out = np.where(np.isfinite(self.value), self.value, x)
        return out, reject

    def describe(self) -> dict:
        """The rule and the counts so far (for a log's meta)."""
        return {"range_dd": [self.lo_dd, self.hi_dd], "step_dd_per_tick": self.step_dd, "cap_ticks": self.cap_ticks,
                "confirm_frames": self.confirm, "frames": int(self.frames), "frames_rejected": int(self.frames_rejected),
                "rejected_per_servo": self.rejected.tolist(), "relocked_per_servo": self.relocked.tolist()}

    def report(self) -> str:
        """One line for the end of a command."""
        r = int(self.rejected.sum())
        s = "feedback filter: %d reading%s rejected in %d of %d STATE frames" % (
            r, "" if r == 1 else "s", self.frames_rejected, self.frames)
        if r:
            s += "; per servo " + " ".join(str(int(v)) for v in self.rejected)
        k = int(self.relocked.sum())
        if k:
            s += "; %d accepted after persisting away (servos %s)" % (
                k, ",".join(str(i + 1) for i in np.flatnonzero(self.relocked)))
        return s


def position(st) -> np.ndarray:
    """A STATE frame's filtered positions (dd, float) when the link filtered it, else its raw ``pos_dd``."""
    f = getattr(st, "pos_filt", None)
    return np.asarray(st.pos_dd if f is None else f, dtype=float)


def filter_log(pos_dd, tick=None, boards=None, **kw) -> tuple[np.ndarray, np.ndarray, FeedbackFilter]:
    """A recorded log through the filter: (filtered N x n float dd, rejected N x n bool, the filter with its counts).
    ``tick``, ``boards``: the log's per-frame arrays, when it has them."""
    pos = np.asarray(pos_dd, dtype=float)
    n = pos.shape[1]
    f = FeedbackFilter(n=n, **kw)
    out = np.empty_like(pos)
    rej = np.zeros(pos.shape, dtype=bool)
    for k in range(len(pos)):
        out[k], rej[k] = f.update(pos[k], None if tick is None else int(tick[k]),
                                  None if boards is None else fresh_joints(boards[k], n))
    return out, rej, f
