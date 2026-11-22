"""Command line for the servo link: ``python3 -m mp4_rl.deploy.mp4link <command>``.

Servo ids 1..12: 1-3 FR, 4-6 FL, 7-9 RR, 10-12 RL (abduction, hip, knee). Positions are the servo's own deci-degrees
(dd, 0.1 deg, 0..3100). Every motion command disarms on exit, and the ESP32 disarms by itself 100 ms after the last frame.

Commands: ``ping`` checks the link; ``config`` writes the servo gains, which every power-off of the servos loses;
``offsets`` reads the calibration; ``dump`` reads every servo parameter; ``gains`` reads the gain registers back
next to what ``config`` and the runner write; ``scale`` reads each servo's range_position_deg; ``watch`` prints the
joints while the limp legs are moved by hand; ``centre`` holds servos at their centre to mount the legs; ``calib``
holds the stand pose and adjusts the offsets, which the ESP32 stores in flash; ``stream`` plays a recorded get-up on
a lifted robot; ``idle`` disarms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import time

import numpy as np
import serial

from ..runner.feedback import position
from . import protocol as p
from .link import DEFAULT_BAUD, DEFAULT_PORT, Link, LinkError

NAMES = ["FR abd", "FR hip", "FR knee", "FL abd", "FL hip", "FL knee",
         "RR abd", "RR hip", "RR knee", "RL abd", "RL hip", "RL knee"]
LEGS = ("FR", "FL", "RR", "RL")
ABDUCTION = (1, 4, 7, 10)
# The gains the policies were trained for; the servos' flash holds softer ones.
GAINS = {"control_mode": 1.0, "kp_position": 80.0, "kd_position": 400.0, "kff_current": 0.001,
         "max_pwm_duty_cycle": 0.99}
# The servo firmware's PWM mode scales the duty with pwm_scale (register 12), not with kff_current: it must read this.
# `config --pwm-scale` and the runner write it; RAM only, so every power-off of the servos loses it.
PWM_SCALE = 0.001
PWM_SCALE_TOL = 0.02           # the runner refuses to arm on a read-back further than this (relative) from PWM_SCALE
MAX_STEP_DEG = 15.0            # the most a target moves per send when holding poses
# The firmware's calibration formula, in float32: stand code = 511 + (int)(offset * 10240 / 3100), and the servo target
# is (uint16)(3100 - code * 3100 / 1024) dd.
CENTRE_TICK = 511
OFFSET_MAX_DEG = 120.0         # stand codes 511 +- 396 (115..907), about 35 deg or more from the track ends (0, 1023); the firmware's limit
ONE_TICK_DEG = 3100.0 / 10240.0
# A joint's potentiometer has a dead zone beyond 0 and 3100 dd.
TRACK_END_DEG = 15.0           # a joint measured this close to an end stops a stream
STREAM_END_MARGIN_DEG = 20.0   # a stream with a target this close to an end is refused
WATCH_NEAR_END_DEG = 5.0       # `watch` flags a reading this close to an end
WATCH_SAVE_S = 10.0            # `watch` rewrites its log this often, so a crash loses at most this much
# A stream plays a get-up: every frame in the firmware's get-up mode, in the get-up policy's band and step limit.
STREAM_ABD_BAND_DEG, STREAM_HIP_KNEE_BAND_DEG = 90.0, 180.0
STREAM_STEP_DEG = 20.0
STREAM_GAP_S = 0.1             # no STATE this long stops a stream


# ---- helpers ------------------------------------------------------------------
class Pacer:
    """Wall-clock pacing: sleep, then spin the last millisecond."""

    def __init__(self, period: float):
        self.period = period
        self.next = time.monotonic()
        self.stamps: list[float] = []

    def wait(self) -> None:
        self.next += self.period
        delay = self.next - time.monotonic()
        if delay > 0.002:
            time.sleep(delay - 0.001)
        while time.monotonic() < self.next:
            pass
        if time.monotonic() - self.next > self.period:   # overrun: restart the grid
            self.next = time.monotonic()
        self.stamps.append(time.monotonic())


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids += list(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    if any(i < 1 or i > 12 for i in ids):
        raise SystemExit("servo ids are 1..12")
    return ids


def offset_ticks(offsets_deg) -> np.ndarray:
    off = np.asarray(offsets_deg, dtype=np.float32)
    return CENTRE_TICK + np.trunc(off * np.float32(10240.0) / np.float32(3100.0)).astype(int)


def ticks_to_dd(ticks) -> np.ndarray:
    t = np.asarray(ticks, dtype=np.float32)
    return np.trunc(np.float32(3100.0) - t * np.float32(3100.0) / np.float32(1024.0)).astype(int)


def read_offsets(link: Link) -> tuple[np.ndarray, list[bool]]:
    """The twelve stored offsets [deg] (unset reads 0) and which are stored."""
    vals = [link.get_offset(i) for i in range(1, 13)]
    return np.array([0.0 if v is None else v for v in vals]), [v is not None for v in vals]


def stand_from_robot(link: Link) -> np.ndarray:
    """The stand pose [dd] from the offsets stored on the robot."""
    return ticks_to_dd(offset_ticks(read_offsets(link)[0])).astype(float)


def require_config(link: Link, ids) -> None:
    """Holding a pose on the servos' flash gains is several times softer."""
    for i in ids:
        kff = link.get_param(i, "kff_current")
        if kff < 0.0009:
            raise LinkError("servo %d runs kff %g, not 0.001: run `config` first" % (i, kff))


def mask(ids) -> int:
    m = 0
    for i in ids:
        m |= 1 << (i - 1)
    return m


def present(link: Link) -> np.ndarray:
    """Positions (filtered, ``feedback.position``) from two consecutive STATE frames with fresh feedback that agree."""
    states = []
    for _ in range(20):
        st = link.wait_state(1.0)
        if st.boards == 0x0F:
            states.append(st)
            if len(states) == 2:
                break
    if len(states) < 2:
        raise LinkError("no fresh feedback from every board")
    a, b = (position(s) for s in states)
    if np.any(a < 100) or np.any(a > 3000) or np.any(np.abs(a - b) > 20):
        raise LinkError("implausible positions (dd): %s / %s" % (a.astype(int).tolist(), b.astype(int).tolist()))
    return b


def clip_band(stand: np.ndarray, target: np.ndarray, band_deg) -> np.ndarray:
    return np.clip(target, stand - np.asarray(band_deg, dtype=float) * 10, stand + np.asarray(band_deg, dtype=float) * 10)


def preflight(link: Link, clear: bool) -> None:
    """Refuse to arm over a latched fault unless ``clear``, and while the firmware reports a servo range other than
    310 deg (STATE flag RANGE: it refuses ARM itself)."""
    st = link.wait_state(1.0)
    if st.flags & p.ST_RANGE:
        raise LinkError("a servo reports a range_position_deg other than %.1f deg and the firmware refuses to arm; "
                        "see `scale`" % (p.RANGE_DD / 10))
    latched = st.flags & (p.ST_FAULTS | p.ST_REJECTED)
    if latched:
        what = "%s (servo %d)" % (", ".join(n for b, n in p.FLAG_NAMES.items() if latched & b), st.fault_servo)
        if not clear:
            raise LinkError("fault latched: %s; look at it, then rerun with --clear" % what)
        print("clearing latched fault:", what)


class Mover:
    """Sends targets, arms on the first frame, keeps the step limit, and (once ``stand`` is set) the band about the stand."""

    def __init__(self, link: Link, enable: int, cap: int, clear: bool = False, band_deg=None, getup: bool = False,
                 max_step_deg: float = MAX_STEP_DEG):
        self.link, self.enable, self.cap, self.clear = link, enable, cap, clear
        self.stand: np.ndarray | None = None
        self.band_deg = band_deg
        self.getup = getup
        self.max_step_deg = max_step_deg
        self.last: np.ndarray | None = None
        self.armed = False
        self.first_seq: int | None = None
        self.clipped = 0

    def send(self, target_dd: np.ndarray) -> None:
        t = np.asarray(target_dd, dtype=float).copy()
        if self.stand is not None:
            c = clip_band(self.stand, t, self.band_deg)
            self.clipped += int(np.any(c != t))
            t = c
        if self.last is not None:
            step = self.max_step_deg * 10
            c = np.clip(t, self.last - step, self.last + step)
            self.clipped += int(np.any(c != t))
            t = c
        t = np.clip(np.rint(t), 0, 3100)
        flags = 0 if self.armed else (p.SA_ARM | (p.SA_CLEAR if self.clear else 0))
        seq = self.link.set_all(t.astype(int), self.enable, flags, self.cap, getup=self.getup)
        if not self.armed:
            self.first_seq = seq
        self.armed = True
        self.last = t


def after_our_arm(link: Link, first_seq: int | None, st) -> bool:
    """The STATE answers one of the frames sent since our arming frame (its seq lies from ``first_seq`` to the last
    sent), so its flags already reflect our ARM (and its CLEAR). Before our first frame (``first_seq`` None), every
    STATE counts."""
    if first_seq is None:
        return True
    return (st.seq - first_seq) & 0xFFFF <= (link.last_seq - first_seq) & 0xFFFF


def check_flags(link: Link, mover=None) -> None:
    """Raise if the reader stopped, or if, once we have armed (``mover.first_seq``), a STATE that answers our frames since
    our ARM shows the firmware not armed: it refused the ARM or disarmed, by its own rules. An earlier STATE may still show
    the state before our ARM and is not judged. The message names the flags the STATE reports."""
    link.check()
    st = link.latest
    first = None if mover is None else mover.first_seq
    if st is None or first is None or not after_our_arm(link, first, st):
        return
    if not st.armed:
        raise LinkError("the firmware is not armed: it refused the ARM or disarmed (STATE flags: %s)"
                        % (", ".join(st.flag_names()) or "none"))


def parent_made(path) -> str:
    """``path``, its folder created if missing."""
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    return str(path)


def save_log(link: Link, path: str, pacer: Pacer | None, meta: dict, states: list | None = None,
             extra: dict | None = None, atomic: bool = False, quiet: bool = False) -> None:
    """Every STATE frame recorded (or ``states``) and every SET_ALL sent, with ``meta`` as JSON; ``extra``: more arrays.
    ``pos_dd`` is raw; ``pos_reject`` marks the readings the link's feedback filter rejected, and the meta's
    ``feedback_filter`` holds its rule and counts.
    ``atomic``: written to ``<path>.tmp`` and renamed over ``path``, so the file is always a whole log."""
    states = list(link.record or []) if states is None else states     # the reader thread keeps appending
    fb = getattr(link, "feedback", None)
    if fb is not None:
        meta = dict(meta, feedback_filter=fb.describe())
    sent = list(link.sent)
    arr = lambda f, dt: np.array([f(s) for s in states], dtype=dt)
    path = parent_made(path)
    if atomic and not path.endswith(".npz"):
        path += ".npz"                 # as numpy names it when given a path
    out = path + ".tmp" if atomic else path
    f = open(out, "wb") if atomic else path
    try:
        _savez(f, link, pacer, meta, states, sent, arr, extra)
        if atomic:
            f.flush()
            os.fsync(f.fileno())       # the bytes on disk before the rename: a power cut keeps the last whole log
    except BaseException:
        if atomic:
            f.close()
            if os.path.exists(out):
                os.remove(out)
        raise
    if atomic:
        f.close()
        os.replace(out, path)
    if not quiet:
        print("log: %s (%d STATE, %d SET_ALL)" % (path, len(states), len(sent)))


def _savez(f, link, pacer, meta, states, sent, arr, extra) -> None:
    np.savez_compressed(
        f,
        t_rx=arr(lambda s: s.t_rx, float), tick=arr(lambda s: s.tick, np.uint32),
        t_us=arr(lambda s: s.t_us, np.uint32), seq=arr(lambda s: s.seq, np.uint16),
        flags=arr(lambda s: s.flags, np.uint16), boards=arr(lambda s: s.boards, np.uint8),
        pos_dd=np.array([s.pos_dd for s in states], dtype=np.uint16).reshape(-1, 12),
        pos_reject=np.array([np.zeros(12, bool) if getattr(s, "pos_reject", None) is None else s.pos_reject
                             for s in states], dtype=bool).reshape(-1, 12),
        cur_ma=np.array([s.cur_ma for s in states], dtype=np.int16).reshape(-1, 12),
        temp_dc=np.array([s.temp_dc for s in states], dtype=np.int16).reshape(-1, 12),
        acc_mg=np.array([s.acc_mg for s in states], dtype=np.int16).reshape(-1, 3),
        gyr_ddps=np.array([s.gyr_ddps for s in states], dtype=np.int16).reshape(-1, 3),
        imu_t_us=arr(lambda s: s.imu_t_us, np.uint32),
        sent_t=np.array([s[0] for s in sent], dtype=float),
        sent_seq=np.array([s[1] for s in sent], dtype=np.uint16),
        sent_pos_dd=np.array([np.frombuffer(s[2][10:34], dtype="<u2") for s in sent],
                             dtype=np.uint16).reshape(-1, 12),
        sent_enable=np.array([int.from_bytes(s[2][6:8], "little") for s in sent], dtype=np.uint16),
        loop_t=np.array(pacer.stamps if pacer else [], dtype=float),
        latency=np.array(link.latency, dtype=float),
        meta=json.dumps(meta),
        **(extra or {}),
    )


def summary(link: Link, pacer: Pacer | None) -> None:
    states = link.record or list(link.states)
    print("STATE frames %d, tick gaps %d, CRC errors %d, bytes skipped %d"
          % (link.n_states, link.tick_gaps, link.parser.crc_errors, link.parser.skipped))
    if len(states) > 1:
        dt = states[-1].t_rx - states[0].t_rx
        print("STATE rate %.1f Hz" % ((len(states) - 1) / dt if dt > 0 else 0))
    if pacer and len(pacer.stamps) > 2:
        per = np.diff(pacer.stamps) * 1000
        print("loop period ms: p50 %.2f  p99 %.2f  max %.2f  (n %d)"
              % (np.percentile(per, 50), np.percentile(per, 99), per.max(), len(per)))
    if link.latency:
        lat = np.array(link.latency) * 1000
        print("SET_ALL -> STATE echo ms: p50 %.2f  p95 %.2f  max %.2f  (n %d)"
              % (np.percentile(lat, 50), np.percentile(lat, 95), lat.max(), len(lat)))
    if states:
        cur = np.abs(np.array([s.cur_ma for s in states], dtype=float))
        tmp = np.array([s.temp_dc for s in states], dtype=float) / 10
        flags = 0
        for s in states:
            flags |= s.flags
        print("max |I| mA:", " ".join("%d" % v for v in cur.max(0)))
        print("max temp C:", " ".join("%.1f" % v for v in tmp.max(0)))
        print("flags seen:", ", ".join(n for b, n in p.FLAG_NAMES.items() if flags & b) or "none")
    if getattr(link, "feedback", None) is not None:
        print(link.feedback.report())


def ramp_to(mover: Mover, pacer: Pacer, link: Link, frm: np.ndarray, to: np.ndarray, seconds: float) -> None:
    n = max(1, int(seconds / pacer.period))
    for k in range(n):
        mover.send(frm + (to - frm) * smoothstep((k + 1) / n))
        pacer.wait()
        check_flags(link, mover)


def hold(mover: Mover, pacer: Pacer, link: Link, target: np.ndarray, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        mover.send(target)
        pacer.wait()
        check_flags(link, mover)


def at_track_end(pos_dd):
    """Whether a measured position (or each of an array) lies within TRACK_END_DEG of 0 or 3100 dd. Give it filtered
    positions (``feedback.position``): a single impossible reading must not stop a stream."""
    pos = np.asarray(pos_dd, dtype=float)
    return (pos < TRACK_END_DEG * 10) | (pos > 3100 - TRACK_END_DEG * 10)


def _stdin_fd():
    """stdin's descriptor when it is a terminal (Enter stops), else None."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        pass
    return None


def _enter_pressed(fd) -> bool:
    if fd is None:
        return False
    if select.select([fd], [], [], 0)[0]:
        os.read(fd, 1024)
        return True
    return False


def _lines_typed(fd) -> list[str] | None:
    """The lines typed since the last call (a terminal hands them over at Enter), or None when nothing waits.
    End of input reads as one empty line."""
    if fd is None or not select.select([fd], [], [], 0)[0]:
        return None
    data = os.read(fd, 1024)
    return [s.strip() for s in data.decode(errors="replace").splitlines()] or [""]


# ---- link and servo checks ------------------------------------------------------
def cmd_ping(link: Link, a) -> None:
    link.record = []
    pacer = Pacer(0.02)
    end = time.monotonic() + a.seconds
    pos = [1550] * 12
    while time.monotonic() < end:
        link.set_all(pos, enable=0, flags=0)       # disarmed: only measures the echo
        pacer.wait()
    summary(link, pacer)
    st = link.latest
    if st is None:
        raise LinkError("no STATE received")
    acc = np.array([s.acc_mg for s in link.record], dtype=float) / 1000
    gyr = np.array([s.gyr_ddps for s in link.record], dtype=float) / 10
    print("IMU %s: acc mean %s g (|a| %.3f), gyro mean %s deg/s, std %s"
          % ("ok" if st.flags & p.ST_IMU_OK else "ABSENT", np.round(acc.mean(0), 3),
             np.linalg.norm(acc.mean(0)), np.round(gyr.mean(0), 2), np.round(gyr.std(0), 2)))
    boards = np.array([[s.boards >> b & 1 for b in range(4)] for s in link.record])
    print("fresh feedback per board (FR, FL, RR, RL): %s" % np.round(boards.mean(0), 3))
    for i in range(12):
        print("  %2d %-9s pos %4d dd (%6.1f deg)  I %5d mA  T %5.1f C"
              % (i + 1, NAMES[i], st.pos_dd[i], st.pos_dd[i] / 10, st.cur_ma[i], st.temp_dc[i] / 10))
    if a.log:
        save_log(link, a.log, pacer, {"cmd": "ping", "seconds": a.seconds})


def cmd_dump(link: Link, a) -> None:
    out = {}
    print("%-4s %-9s " % ("id", "") + " ".join("%10s" % n[:10] for n in p.PARAMS) + "  temp_C  supply_mA")
    for i in range(1, 13):
        row = {}
        for name in p.PARAMS:
            try:
                row[name] = link.get_param(i, name)
            except LinkError:
                row[name] = None
        for key, lid in (("temperature_c", p.LIVE_TEMPERATURE_C), ("supply_current_ma", p.LIVE_SUPPLY_CURRENT_MA)):
            try:
                row[key] = link.get_live(i, lid)
            except LinkError:
                row[key] = None
        out[i] = row
        fmt = lambda v: "%10s" % ("-" if v is None else "%.6g" % v)
        print("%-4d %-9s " % (i, NAMES[i - 1]) + " ".join(fmt(row[n]) for n in p.PARAMS)
              + "  %6s  %9s" % (fmt(row["temperature_c"]).strip(), fmt(row["supply_current_ma"]).strip()))
    if a.out:
        with open(parent_made(a.out), "w") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "servos": out}, f, indent=1)
        print("saved", a.out)


def cmd_config(link: Link, a) -> None:
    writes = {**GAINS, "ki_position": 0.0, "kp_current": 0.0}
    if a.pwm_scale is not None:
        writes["pwm_scale"] = a.pwm_scale
    for i in range(1, 13):
        got = []
        for name, value in writes.items():
            rb = link.set_param(i, name, value)
            if abs(rb - value) > 1e-6 + 1e-4 * abs(value):
                raise LinkError("servo %d %s: wrote %g, read back %g" % (i, name, value, rb))
            got.append("%s=%g" % (name, rb))
        print("%2d %-9s %s" % (i, NAMES[i - 1], " ".join(got)))
    print("written to RAM (not saved; a power cycle restores each servo's flash)")


def _read_raw(link: Link, servo: int, pid: int) -> p.Response | None:
    """One GET_PARAM, the response whatever its status (None: no response)."""
    try:
        return link.request(p.CMD_GET_PARAM, servo, pid, check=False)
    except LinkError:
        return None


def cmd_gains(link: Link, a) -> None:
    """Read-only: every servo's parameter registers 0..12 (GET_PARAM only), one row per register, next to what `config`
    and the runner write (GAINS, ki_position 0, kp_current 0, pwm_scale PWM_SCALE: the runner always, `config` with
    --pwm-scale). ``!`` marks a register where a servo differs from that."""
    sends = {**GAINS, "ki_position": 0.0, "kp_current": 0.0, "pwm_scale": PWM_SCALE}
    table: dict[str, list] = {}
    print("%-3s %-23s %10s " % ("id", "register", "writes") + " ".join("%9s" % ("s%d" % i) for i in range(1, 13)))
    for pid, name in enumerate(p.ALL_PARAMS):
        row = [_read_raw(link, i, pid) for i in range(1, 13)]
        table[name] = row
        vals = [r.val if r is not None and r.ok else None for r in row]
        want = sends.get(name)
        off = want is not None and any(v is None or abs(v - want) > 1e-6 + 1e-3 * abs(want) for v in vals)
        cell = (lambda r: "%9s" % ("-" if r is None or not r.ok else "%08x" % r.raw_val)) if a.hex else \
               (lambda r: "%9s" % ("-" if r is None or not r.ok else "%.6g" % r.val))
        print("%-3d %-23s %10s " % (pid, name + (" !" if off else ""), "-" if want is None else "%g" % want)
              + " ".join(cell(r) for r in row))
    print("- = no answer. `!`: a servo differs from what config and the runner write.")
    pwm = [r for r in table["pwm_scale"] if r is not None and r.ok]
    if pwm:
        print("pwm_scale answers on %d servos: a servo firmware whose PWM-mode duty scales with pwm_scale, which must read "
              "%g (`config --pwm-scale %g`; the runner writes it before arming)." % (len(pwm), PWM_SCALE, PWM_SCALE))
    if a.out:
        def js(r):
            return None if r is None else {"status": r.status, "val": r.val, "bits": "%08x" % r.raw_val}
        with open(parent_made(a.out), "w") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "writes": sends,
                       "registers": {name: {str(i + 1): js(r) for i, r in enumerate(row)} for name, row in table.items()}},
                      f, indent=1)
        print("saved", a.out)


def cmd_scale(link: Link, a) -> None:
    """Read-only: each servo's range_position_deg (GET_PARAM id 3, which also refreshes the firmware's range check), per
    board, against the 3100 +- 1 dd everything here assumes."""
    bad, answered = [], 0
    for b, leg in enumerate(LEGS):
        cells = []
        for i in range(3 * b + 1, 3 * b + 4):
            r = _read_raw(link, i, p.RANGE_PARAM)
            if r is None or not r.ok:
                cells.append("%2d %-9s  no answer" % (i, NAMES[i - 1]))
                continue
            answered += 1
            dd = r.val * 10.0
            ok = abs(dd - p.RANGE_DD) <= p.RANGE_TOL_DD + 1e-3
            if not ok:
                bad.append(i)
            cells.append("%2d %-9s %7.2f deg = %7.1f dd %s" % (i, NAMES[i - 1], r.val, dd, "ok" if ok else "MISMATCH"))
        print("board %d %s: %s" % (b, leg, "; ".join(cells)))
    st = link.wait_state(1.0)
    print("firmware: %s" % ("RANGE set, ARM refused" if st.flags & p.ST_RANGE else "no RANGE flag"))
    if bad:
        raise LinkError("servos %s report a range other than %d +- %d dd: positions here assume %d"
                        % (bad, p.RANGE_DD, p.RANGE_TOL_DD, p.RANGE_DD))
    if not answered:
        raise LinkError("no servo answered a range read")


def cmd_idle(link: Link, a) -> None:
    link.disarm()
    st = link.wait_state(0.5)
    print("flags:", ", ".join(st.flag_names()) or "none")


# ---- playing a get-up on a lifted robot --------------------------------------------
def stream_band() -> np.ndarray:
    return np.array([STREAM_ABD_BAND_DEG if i + 1 in ABDUCTION else STREAM_HIP_KNEE_BAND_DEG for i in range(12)])


def check_track_ends(stand: np.ndarray, delta: np.ndarray, band) -> None:
    """Refuse a stream whose targets or stand come within STREAM_END_MARGIN_DEG of a track end."""
    tgt = clip_band(stand, stand + delta, band)
    lo_dd, hi_dd = STREAM_END_MARGIN_DEG * 10, 3100 - STREAM_END_MARGIN_DEG * 10
    for i in range(12):
        dmin, dmax = min(float(tgt[:, i].min()), stand[i]), max(float(tgt[:, i].max()), stand[i])
        if lo_dd <= dmin and dmax <= hi_dd:
            continue
        bad = dmin if dmin < lo_dd else dmax
        raise SystemExit("stream: servo %d (%s) reaches %d dd (%+.1f deg from the stand %d dd), within %g deg of "
                         "the track end (0..3100 dd)"
                         % (i + 1, NAMES[i], round(bad), (bad - stand[i]) / 10, stand[i], STREAM_END_MARGIN_DEG))


def cmd_stream(link: Link, a) -> None:
    """Ramp to the stream's first pose, hold 1 s, play the file, ramp back to the stand, hold 1 s, disarm. Any stop
    returns to the stand the same way."""
    band = stream_band()
    stand = stand_from_robot(link)
    data = np.load(a.file)
    delta = np.asarray(data["delta_dd"], dtype=float)
    period = float(data["period_s"])
    if delta.ndim != 2 or delta.shape[1] != 12 or len(delta) == 0 or not np.all(np.isfinite(delta)):
        raise SystemExit("%s: delta_dd must be N x 12 finite values" % a.file)
    check_track_ends(stand, delta, band)
    link.record = []
    start = present(link)
    preflight(link, a.clear)
    fd = _stdin_fd()
    mover = Mover(link, 0x0FFF, a.cap, a.clear, band_deg=band, getup=True, max_step_deg=STREAM_STEP_DEG)
    pacer = Pacer(0.02)
    first = clip_band(stand, stand + delta[0], band)
    reason, t_play = None, [None, None]
    idx, end_hit = 0, None

    def stop_reason() -> str | None:
        st = link.latest
        if time.monotonic() - st.t_rx > STREAM_GAP_S:
            return "state_gap"
        if _enter_pressed(fd):
            return "enter"
        return None

    def track_end() -> str | None:
        """Every STATE frame since the last call, during the playback only."""
        nonlocal idx, end_hit
        new = link.record[idx:]
        idx += len(new)
        for s in new:
            if s.boards != 0x0F:              # a board without fresh feedback: its positions may be a boot's zeros
                continue
            q = position(s)
            hit = np.flatnonzero(at_track_end(q))
            if len(hit):
                end_hit = (int(hit[0]) + 1, int(round(q[hit[0]])))
                return "track_end"
        return None

    try:
        ramp_to(mover, pacer, link, start, first, 2.0)     # from wherever the legs are: only its end is clipped
        mover.stand = stand
        end = time.monotonic() + 1.0
        while reason is None and time.monotonic() < end:
            mover.send(first)
            pacer.wait()
            check_flags(link, mover)
            reason = stop_reason()
        pacer.period = period
        t_play[0] = time.monotonic()
        idx = len(link.record)
        for row in delta:
            if reason is not None:
                break
            mover.send(stand + row)
            pacer.wait()
            check_flags(link, mover)
            reason = stop_reason() or track_end()
        t_play[1] = time.monotonic()
        pacer.period = 0.02
        ramp_to(mover, pacer, link, mover.last, stand, 1.0)
        hold(mover, pacer, link, stand, 1.0)
    finally:
        link.disarm()
        summary(link, pacer)
        states = list(link.record or [])
        play = [s for s in states if t_play[0] is not None and t_play[0] <= s.t_rx <= (t_play[1] or math.inf)]
        if play:
            cur = np.abs(np.array([s.cur_ma for s in play], dtype=float))
            print("playback p99 |I| mA:", " ".join("%d" % v for v in np.percentile(cur, 99, axis=0)))
        if end_hit is not None:
            print("stream: servo %d (%s) measured at %d dd, within %g deg of a track end: stopped and returned"
                  % (end_hit[0], NAMES[end_hit[0] - 1], end_hit[1], TRACK_END_DEG), flush=True)
        print("stream %s; targets clipped on %d sends" % (reason or "completed", mover.clipped), flush=True)
        if a.log:
            save_log(link, a.log, pacer,
                     {"cmd": "stream", "file": a.file, "period_s": period, "stand": stand.tolist(), "cap": a.cap,
                      "band_deg": band.tolist(), "stop_reason": reason or "completed", "clipped_sends": mover.clipped,
                      "t_play": t_play,
                      "max_step_deg": mover.max_step_deg, "track_end_deg": TRACK_END_DEG, "track_end_servo": end_hit},
                     states=states)


# ---- watching the limp legs ----------------------------------------------------
def _fmt(v, f: str) -> str:
    return "-" if v is None else f % v


def cmd_watch(link: Link, a) -> None:
    """Prints the chosen joints once a second as degrees from the stand, with their running min/max, both from the
    filtered positions (``feedback.position``; a rejected raw reading is shown beside them). The log is rewritten
    every WATCH_SAVE_S and at the end, each time whole (a temporary file renamed over it). ``m`` + Enter records a marker
    (the time and the index of the latest STATE frame, saved as ``markers``); Enter alone or ``q`` + Enter ends."""
    ids = list(range(1, 13)) if a.leg is None else [LEGS.index(a.leg) * 3 + n for n in (1, 2, 3)]
    ref = stand_from_robot(link)
    log_path = a.log or "watch_%s.npz" % time.strftime("%Y%m%d_%H%M%S")
    fd = _stdin_fd()
    link.record = []
    pacer = Pacer(0.02)
    pos = [1550] * 12
    flags = p.SA_DISARM if a.disarm else 0
    lo = np.full(12, np.inf)
    hi = np.full(12, -np.inf)
    near = WATCH_NEAR_END_DEG * 10
    markers: list[tuple[float, int]] = []      # (time.monotonic, index of the latest STATE frame)
    print("watch %s (servos idle%s) for %g s, degrees from the stand; m + Enter marks, Enter or q + Enter ends, "
          "Ctrl-C ends and saves" % (a.leg or "all legs", ", DISARM on every frame" if a.disarm else "", a.seconds),
          flush=True)
    reason, idx = "time", 0
    t0 = time.monotonic()
    last_print = last_save = t0

    def save(stop_reason: str, quiet: bool) -> None:
        seen = np.isfinite(lo)
        mn = [float((lo[i] - ref[i]) / 10) if seen[i] else None for i in range(12)]
        mx = [float((hi[i] - ref[i]) / 10) if seen[i] else None for i in range(12)]
        save_log(link, log_path, pacer, {"cmd": "watch", "seconds": a.seconds, "leg": a.leg, "stand": ref.tolist(),
                                         "disarm": bool(a.disarm), "stop_reason": stop_reason,
                                         "min_deg": mn, "max_deg": mx,
                                         "min_dd": [float(v) if seen[i] else None for i, v in enumerate(lo)],
                                         "max_dd": [float(v) if seen[i] else None for i, v in enumerate(hi)],
                                         "near_end_deg": WATCH_NEAR_END_DEG,
                                         "markers": "rows of (time.monotonic s, the clock of t_rx; index of the "
                                                    "latest STATE frame, -1 before the first)",
                                         "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                 extra={"markers": np.array(markers, dtype=float).reshape(-1, 2)}, atomic=True, quiet=quiet)

    try:
        while time.monotonic() - t0 < a.seconds:
            link.set_all(pos, enable=0, flags=flags)
            pacer.wait()
            link.check()
            new = link.record[idx:]
            idx += len(new)
            if new:
                arr = np.array([position(s) for s in new], dtype=float).reshape(-1, 12)
                lo, hi = np.minimum(lo, arr.min(0)), np.maximum(hi, arr.max(0))
            if time.monotonic() - last_print >= 1.0 and link.latest is not None:
                last_print = time.monotonic()
                st = link.latest
                print("t %.0f s" % (last_print - t0), flush=True)
                qs = position(st)
                rej = st.pos_reject if st.pos_reject is not None else np.zeros(12, bool)
                for i in ids:
                    q = float(qs[i - 1])
                    print("  %2d %-9s %+7.1f deg  min %+7.1f  max %+7.1f%s%s"
                          % (i, NAMES[i - 1], (q - ref[i - 1]) / 10, (lo[i - 1] - ref[i - 1]) / 10,
                             (hi[i - 1] - ref[i - 1]) / 10,
                             "  near track end (%d dd)" % q if q < near or q > 3100 - near else "",
                             "  (raw %d dd rejected)" % st.pos_dd[i - 1] if rej[i - 1] else ""), flush=True)
            lines = _lines_typed(fd)
            for line in lines or []:
                if line.lower() == "m":
                    markers.append((time.monotonic(), len(link.record) - 1))
                    print("marker %d at t %.1f s (STATE frame %d)" % (len(markers), markers[-1][0] - t0, markers[-1][1]),
                          flush=True)
                else:
                    reason = "enter" if line.lower() != "q" else "q"
                    break
            if reason != "time":
                break
            if time.monotonic() - last_save >= WATCH_SAVE_S:
                save("running", quiet=True)
                last_save = time.monotonic()
    except KeyboardInterrupt:
        reason = "ctrl-c"
    finally:
        seen = np.isfinite(lo)
        print("watch ended (%s); min / max deg from the stand:" % reason)
        for i in ids:
            mn = float((lo[i - 1] - ref[i - 1]) / 10) if seen[i - 1] else None
            mx = float((hi[i - 1] - ref[i - 1]) / 10) if seen[i - 1] else None
            print("  %2d %-9s min %s  max %s" % (i, NAMES[i - 1], _fmt(mn, "%+7.1f"), _fmt(mx, "%+7.1f")))
        save(reason, quiet=False)
        if getattr(link, "feedback", None) is not None:
            print(link.feedback.report())
        if markers:
            print("%d marker%s" % (len(markers), "s" if len(markers) > 1 else ""))


# ---- calibration --------------------------------------------------------------
def cmd_offsets(link: Link, a) -> None:
    offs, stored = read_offsets(link)
    ticks = offset_ticks(offs)
    dd = ticks_to_dd(ticks)
    for i in range(12):
        off = ("%+6.1f deg" % offs[i]) if stored[i] else "     unset"
        print("%2d %-9s offset %s   stand code %4d   %4d dd" % (i + 1, NAMES[i], off, ticks[i], dd[i]))
    if a.json:
        with open(parent_made(a.json), "w") as f:
            json.dump({"stand_dd": dd.tolist(), "stand_ticks": ticks.tolist(), "offsets_deg": offs.tolist(),
                       "offsets_stored": stored,
                       "source": "the robot's stored offsets, read %s; code = 511 + (int)(offset*10240/3100), "
                                 "dd = (uint16)(3100 - code*3100/1024)" % time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=1)
        print("written:", a.json)


def _status(link: Link, ids) -> str:
    st = link.latest
    k = np.array(ids) - 1
    return "max |I| %d mA, max housing %.1f C" % (np.abs(st.cur_ma[k]).max(), st.temp_dc[k].max() / 10.0)


def cmd_centre(link: Link, a) -> None:
    ids = parse_ids(a.ids)
    require_config(link, ids)
    start = present(link)
    preflight(link, a.clear)
    target = start.copy()
    target[np.array(ids) - 1] = ticks_to_dd([CENTRE_TICK])[0]
    mover = Mover(link, mask(ids), a.cap, a.clear)
    pacer = Pacer(0.02)
    print("servos %s to the centre: code %d = %d dd, offsets ignored. Ctrl-C ends (all idle)."
          % (a.ids, CENTRE_TICK, ticks_to_dd([CENTRE_TICK])[0]))
    try:
        ramp_to(mover, pacer, link, start, target, 2.0)
        end, last = time.monotonic() + a.seconds, 0.0
        while time.monotonic() < end:
            mover.send(target)
            pacer.wait()
            check_flags(link, mover)
            if time.monotonic() - last > 5.0:
                last = time.monotonic()
                print("  holding, %s, %.0f s left" % (_status(link, ids), end - last))
    except KeyboardInterrupt:
        pass
    finally:
        link.disarm()
    print("idle")


def _keys(fd):
    """Characters typed since the last call (non-blocking)."""
    out = ""
    while select.select([fd], [], [], 0)[0]:
        out += sys.stdin.read(1)
    return out


def cmd_calib(link: Link, a) -> None:
    offs, stored = read_offsets(link)
    if a.zero:
        offs[np.array(parse_ids(a.zero)) - 1] = 0.0
    start_offs = offs.copy()
    for item in a.set or []:
        sid, deg = item.split("=")
        offs[int(sid) - 1] = float(deg)
    if np.any(np.abs(offs) > OFFSET_MAX_DEG):
        raise SystemExit("offsets are limited to +-%g deg" % OFFSET_MAX_DEG)
    interactive = a.seconds is None
    if interactive and not sys.stdin.isatty():
        raise SystemExit("calib needs a terminal (ssh -t), or --seconds for a timed hold")
    if interactive:
        import termios
        import tty
    require_config(link, range(1, 13))
    pose = lambda: ticks_to_dd(offset_ticks(offs)).astype(float)
    if a.log:
        link.record = []
    start = present(link)
    preflight(link, a.clear)
    mover = Mover(link, 0x0FFF, a.cap, a.clear)
    pacer = Pacer(0.02)
    save = a.save
    fd = sys.stdin.fileno() if interactive else None
    old = termios.tcgetattr(fd) if interactive else None
    try:
        ramp_to(mover, pacer, link, start, pose(), a.ramp_s)
        if not interactive:
            hold(mover, pacer, link, pose(), a.seconds)
        else:
            tty.setcbreak(fd)
            sel, step = 0, a.step
            print("Stand pose held (code 511 + offset). Keys: ] / [ next / previous servo, + / - step,\n"
                  "f fine (one code step, %.2f deg) / coarse, r restore this servo, s save and quit,\n"
                  "q quit without saving (Ctrl-C too)." % ONE_TICK_DEG)

            def show():
                print("  servo %2d %-9s offset %+6.2f deg   stand code %4d   step %.2f deg   [%s]"
                      % (sel + 1, NAMES[sel], offs[sel], offset_ticks(offs)[sel], step, _status(link, range(1, 13))))
            show()
            done = False
            while not done:
                for c in _keys(fd):
                    if c in "]n":
                        sel = (sel + 1) % 12
                    elif c in "[p":
                        sel = (sel - 1) % 12
                    elif c in "+=":
                        offs[sel] = min(offs[sel] + step, OFFSET_MAX_DEG)
                    elif c in "-_":
                        offs[sel] = max(offs[sel] - step, -OFFSET_MAX_DEG)
                    elif c == "f":
                        step = ONE_TICK_DEG if step != ONE_TICK_DEG else a.step
                    elif c == "r":
                        offs[sel] = start_offs[sel]
                    elif c == "s":
                        save, done = True, True
                    elif c == "q":
                        save, done = False, True
                    else:
                        continue
                    show()
                    if done:
                        break
                mover.send(pose())
                pacer.wait()
                check_flags(link, mover)
    except KeyboardInterrupt:
        save = False if interactive else save
    finally:
        if interactive:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        link.disarm()
        if a.log:
            save_log(link, a.log, pacer, {"cmd": "calib", "cap": a.cap, "ramp_s": a.ramp_s, "seconds": a.seconds,
                                          "start_offsets_deg": start_offs.tolist(), "offsets_deg": offs.tolist(),
                                          "save": bool(save), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    for i in range(12):
        flag = "" if offs[i] == start_offs[i] else "   (was %+.2f)" % start_offs[i]
        print("%2d %-9s offset %+6.2f deg%s" % (i + 1, NAMES[i], offs[i], flag))
    if not save:
        print("not saved: the robot keeps the previous offsets")
        return
    t_end = time.monotonic() + 1.0                       # offsets are refused while armed
    while link.wait_state(0.5).armed and time.monotonic() < t_end:
        pass
    for i in range(12):
        link.set_offset(i + 1, float(offs[i]))
    n = link.save_offsets()
    back, _ = read_offsets(link)
    if not np.allclose(back, offs.astype(np.float32), atol=1e-4):
        raise LinkError("read-back differs: %s" % back.tolist())
    print("saved %d offsets and read them back" % n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mp4_rl.deploy.mp4link", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT, help="serial port (default %(default)s)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="default %(default)s")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ping", help="link check: STATE rate, CRC, IMU, latency (servos stay idle)")
    s.add_argument("--seconds", type=float, default=10.0, help="default %(default)s")
    s.add_argument("--log", help="save the STATE stream and the frames sent to this .npz")
    s = sub.add_parser("watch", help="servos idle: print where the joints are while the legs are moved by hand")
    s.add_argument("--seconds", type=float, default=60.0, help="default %(default)s")
    s.add_argument("--log", help="the .npz log (default watch_<time>.npz here)")
    s.add_argument("--leg", choices=LEGS, help="print only this leg's joints (default all)")
    s.add_argument("--no-disarm", dest="disarm", action="store_false",
                   help="send frames without DISARM (by default every frame carries DISARM: firmware that holds the pose "
                        "on plain frames lets the legs go limp only then)")
    s = sub.add_parser("dump", help="read every servo parameter (read-only)")
    s.add_argument("--out", help="also write them to this JSON file")
    s = sub.add_parser("config", help="write the gains to the servos' RAM and read them back")
    s.add_argument("--pwm-scale", type=float, metavar="VALUE",
                   help="also write pwm_scale (register 12; the servo firmware's PWM duty scale, %g) and read it back"
                        % PWM_SCALE)
    s = sub.add_parser("gains", help="read every servo's registers 0..12 next to what config writes (read-only)")
    s.add_argument("--hex", action="store_true", help="the raw 32-bit words instead of the floats")
    s.add_argument("--out", help="also write them to this JSON file")
    sub.add_parser("scale", help="read each servo's range_position_deg against 3100 dd (read-only)")
    s = sub.add_parser("stream", help="play a get-up on a lifted robot, in the firmware's get-up mode")
    s.add_argument("file", help="npz with delta_dd (N x 12, targets minus the stand) and period_s")
    s.add_argument("--cap", type=int, default=600,
                   help="frame cap mA; in control mode 1 a duty cap of kff x cap (default %(default)s: duty 0.6)")
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    s.add_argument("--log", help="the .npz log")
    s = sub.add_parser("offsets", help="read the calibration offsets stored on the robot")
    s.add_argument("--json", help="also write them, with the stand they give, to this file")
    s = sub.add_parser("centre", help="hold servos at the centre (code 511, offsets ignored) to mount the legs")
    s.add_argument("--ids", default="1-12")
    s.add_argument("--cap", type=int, default=300, help="frame cap mA (default %(default)s)")
    s.add_argument("--seconds", type=float, default=600.0)
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    s = sub.add_parser("calib", help="hold the stand pose, adjust the offsets, save them on the robot")
    s.add_argument("--step", type=float, default=1.0, help="deg per key press (f: one code step)")
    s.add_argument("--cap", type=int, default=300, help="frame cap mA (default %(default)s)")
    s.add_argument("--zero", metavar="IDS", help="start these servos from offset 0 (after `centre` and mounting the leg)")
    s.add_argument("--set", action="append", metavar="ID=DEG", help="start from this offset (repeatable)")
    s.add_argument("--seconds", type=float, help="no keys: hold this long, then stop")
    s.add_argument("--save", action="store_true", help="with --seconds: save the offsets after")
    s.add_argument("--ramp-s", type=float, default=2.0, metavar="S",
                   help="s of the ramp from where the legs are to the stand (default %(default)s)")
    s.add_argument("--log", help="save the STATE stream and the frames sent to this .npz (also on q or Ctrl-C)")
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    sub.add_parser("idle", help="disarm: every servo idle")

    a = ap.parse_args(argv)
    try:
        link = Link(a.port, a.baud)
    except serial.SerialException as exc:
        print("error: cannot open %s (%s)" % (a.port, exc), file=sys.stderr)
        return 1
    try:
        {"ping": cmd_ping, "dump": cmd_dump, "config": cmd_config, "stream": cmd_stream, "idle": cmd_idle,
         "offsets": cmd_offsets, "centre": cmd_centre, "calib": cmd_calib, "watch": cmd_watch, "gains": cmd_gains,
         "scale": cmd_scale}[a.cmd](link, a)
    except LinkError as exc:
        print("error:", exc, file=sys.stderr)
        return 1
    finally:
        link.close()
    return 0
