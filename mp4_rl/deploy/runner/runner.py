"""Run the walking and get-up policies on a Mini Pupper v4 (SKU2) from its Raspberry Pi 5.

    python3 -m mp4_rl.deploy.runner check [--policy DIR]
    python3 -m mp4_rl.deploy.runner imu --seconds 25
    python3 -m mp4_rl.deploy.runner run --vx 0 --seconds 40 --log logs/run1

The walking policy stands and walks. When the robot falls, the get-up policy takes over; once the robot is standing the
walking policy takes over again.

**The loop.** One policy step per 4 STATE frames (20 ms, locked to the ESP32's clock). Every frame goes into the velocity
filter, the fall detector and the heading loop, in tick order. Each step builds the observation from the newest frame, runs
the policy, decodes its action into targets (``common.policy.ActionDecoder``: clip, scale, band, 20 deg step limit, joint limits,
the firmware's ticks) and sends one SET_ALL.

**The sequence.** One second with the servos idle measures the gyro's bias: keep the robot still (a bias over
``GYRO_BIAS_MAX`` on any axis refuses the run: the robot moved). A ramp to the walking
policy's stand follows, then the policy loop. With ``--start-mode getup`` (the robot placed lying down) there is no ramp: the
loop starts in the get-up.

**A fall** (``estimators.FallDetector`` on the complementary-filter tilt) during the ramp or walking switches to the get-up
policy at the next step: its decoder seeded from the measured pose, the observation rebuilt, the command inputs carrying its
phase clock, and SA_GETUP on every SET_ALL (the firmware's get-up mode). The firmware must echo the mode (ST_GETUP) within
0.2 s; a firmware that does not enter it does not support the get-up, and the run stops (limp).
**A retry:** when the phase has been at its end for ``--getup-phase-restart`` s and the tilt is still above 45 deg, the clock
restarts. **The hand-back:** tilt < 12 deg, |gyro| < 0.8 rad/s and the pose within 10 deg rms of the stand (``--handback-tilt``,
``--handback-gyro``, ``--handback-rms``), held for 25 steps, returns to the walking policy, its decoder seeded from the measured
pose and the heading re-latched. Not up after
``--getup-timeout`` s, or one fall more than ``--getup-max``: limp, and the run ends.

**Stops.** The end of ``--seconds``, Enter, an invalid IMU frame, a non-finite action, a STATE gap or two late steps in a
row: ramp to the stand, hold, disarm; during a get-up, limp at once (no ramp from a fallen pose). Ctrl-C, a
link fault or a target that fails its check: limp at once. A latched fault refuses the next launch until ``--clear``.

**The first step.** The firmware disarms when frames stop for 100 ms, and between the ramp's last send and the first step
nothing else would be sent. So one step of each policy runs before arming (``Runner.warm_up``: first calls are slow, and hold
the interpreter), and until the first step the last target is repeated on every STATE frame (``Runner.keep_fed``).

**Calibration.** Each policy folder carries the calibration it was exported for (``sku2_contract.json``); the runner
reads it from there and refuses a folder without one, a calibration the policy was not exported for, and a walking and
a get-up policy with different calibrations (``checks.folder_calibration``, ``checks.calibration_file_refusal``,
``checks.pair_calibration_refusal``).

**Refusals before arming.** A latched fault, a servo gain that does not read back (pwm_scale, register 12: within 2 % of
0.001), no valid IMU, a joint reading within 10 deg of a track end, a policy trained on another servo model or robot model
(``checks.robot_refusal``), a policy whose bounds were made for another calibration (``checks.band_refusal``), a decoder
that can reach a target too near a track end (``checks.decoder_selftest``), a gyro bias over ``GYRO_BIAS_MAX``. A
difference between the robot's offsets and the contract's is printed and logged.

**The firmware's safeguards are the firmware's.** It decides by its own rules when to refuse an ARM or to disarm; the
runner stops, limp, at the first STATE after its arming frame that shows the firmware not armed, and names the STATE
flags (``cli.check_flags``).

**The reading watch (it prints and logs; nothing stops).** At the start and during a get-up, a position reading within
5 deg of a track end.

**Position feedback.** Every STATE frame's joint readings go through the link's plausibility filter
(``runner/feedback.py``): the observation, the hand-back, the decoder's restarts and the reading watch use the filtered
positions; the logs keep the raw ones with a ``pos_reject`` mask; the run ends by printing the rejections.

**Logs.** ``<log>.npz``: every STATE and SET_ALL. ``<log>.policy.npz``: per step the STATE tick used, times, observation, raw
action, target, command, ``mode`` (0 walk, 1 get-up), filtered tilt and |gyro|; its meta lists every get-up (``getups``: entry,
trigger, restarts, end). ``<log>.stand.json``: the stand and the robot's offsets. Logs are written at the end of the run.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mp4_rl import paths
from mp4_rl.common.policy import (
    CONTROL_DT,
    TRACK_DD,
    ActionDecoder,
    Contract,
    HistoryStacker,
    has_heading_obs,
    obs_size,
    policy_band_deg,
    policy_bounds,
    policy_filter_alpha,
    policy_kind,
    policy_stand_offset_deg,
)
from mp4_rl.common.policy_file import NumpyPolicy, fixture_tolerance
from mp4_rl.deploy.mp4link import protocol as p
from mp4_rl.deploy.mp4link.cli import GAINS as IDENTIFIED
from mp4_rl.deploy.mp4link.cli import (
    PWM_SCALE,
    PWM_SCALE_TOL,
    Pacer,
    check_flags,
    preflight,
    present,
    read_offsets,
    save_log,
    smoothstep,
)
from mp4_rl.deploy.runner.checks import (
    TARGET_END_DEG,
    band_refusal,
    calibration_file_refusal,
    decoder_selftest,
    folder_calibration,
    pair_calibration_refusal,
    robot_refusal,
    target_refusal,
)
from mp4_rl.deploy.runner.estimators import (  # noqa: F401
    ACC_DROPOUT_MG,
    CF_GATE_G,
    CF_TAU_S,
    DEG,
    FALL_FAST_GYRO,
    FALL_FAST_TILT_DEG,
    FALL_HOLD_S,
    FALL_TILT_DEG,
    FallDetector,
    GravityFilter,
    HeadingLoop,
    ObservationBuilder,
    getup_phase,
    imu_valid,
    tilt_deg,
)
from mp4_rl.deploy.runner.feedback import position

TICKS_PER_STEP = int(round(CONTROL_DT / 0.005))
# The gyro's bias, measured with the robot still before each run, beyond which the run is refused (rad/s, any axis). On
# the robot, 28 still starts read at most 0.0099 rad/s on any axis; one where the robot was moved during the second read
# 0.054 and 0.082.
GYRO_BIAS_MAX = 0.02
CAP_MA = 2000                  # the frame's cap: in the servos' control mode 1, duty <= min(max_pwm, kff x cap)
HOLD_AFTER_S = 5.0             # at the stand before disarming at the end of a run
HANDBACK_TILT_DEG, HANDBACK_GYRO, HANDBACK_RMS_DEG, HANDBACK_STEPS = 12.0, 0.8, 10.0, 25
ECHO_TIMEOUT_S = 0.2
PHASE_RESTART_TILT_DEG = 45.0
READING_WATCH_DEG = 5.0
READING_WATCH_DD = READING_WATCH_DEG * 10.0
MODE_WALK, MODE_GETUP = 0, 1

# The IMU's mounting, sensor -> body. imu_mount.json (the robot's measured data in mp4_sim; the deployment copies it
# beside this file) names the one measured ({"mount": name}), or gives it as
# {"mount": "custom", "R_sensor_to_body": 3 x 3}; `imu` prints gravity in each candidate (and the custom one).
IMU_MOUNTS = {
    "flip_x": np.diag([1.0, -1.0, -1.0]),
    "flip_y": np.diag([-1.0, 1.0, -1.0]),
    "swap_xy": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    "swap_xy_neg": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
}


def _mount_file() -> Path:
    """``imu_mount.json`` beside this file (a deployment), else the installed ``mp4_sim``'s SKU2 file."""
    here = Path(__file__).resolve().parent / "imu_mount.json"
    if here.is_file():
        return here
    import mp4_sim
    return Path(mp4_sim.__file__).resolve().parent / "robot" / "assets" / "sku_2" / "measured" / "imu_mount.json"


IMU_MOUNT_JSON = _mount_file()


class StopRun(Exception):
    """A graceful stop: ramp to the stand, hold, disarm."""


class Disarm(Exception):
    """An immediate stop: limp now."""


OFFSET_REFUSE_DEG = 1.0                            # robot vs contract offsets: refuse to arm beyond this
LATE_STEPS_STOP = 2                                # consecutive late policy steps that stop a run


class FallDetected(Exception):
    """A fall during the stand ramp or hold: the policy loop starts in the get-up."""


def mount_rotation(m) -> np.ndarray:
    """``m`` as a sensor -> body mounting: a 3 x 3 proper rotation (orthonormal, det +1) with entries in {-1, 0, 1}."""
    R = np.asarray(m, dtype=float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError(f"R_sensor_to_body must be 3 x 3, not {R.tolist()}")
    if not np.all(np.isin(R, (-1.0, 0.0, 1.0))):
        raise ValueError(f"R_sensor_to_body entries must be -1, 0 or 1: {R.tolist()}")
    if not np.array_equal(R @ R.T, np.eye(3)) or round(float(np.linalg.det(R))) != 1:
        raise ValueError(f"R_sensor_to_body is not a proper rotation (orthonormal, det +1): {R.tolist()}")
    return R


def load_mount(name: str | None = None, path: Path | None = None) -> tuple[str, np.ndarray]:
    """A mounting by name (``custom``: the matrix in ``path``); by default the measured one, in ``path``
    (``IMU_MOUNT_JSON``)."""
    spec = json.loads((path or IMU_MOUNT_JSON).read_text()) if name in (None, "custom") else {}
    name = name or spec["mount"]
    if name == "custom":
        return name, mount_rotation(spec.get("R_sensor_to_body"))
    if name not in IMU_MOUNTS:
        raise ValueError(f"unknown IMU mount {name!r}: one of {sorted(IMU_MOUNTS)} or custom")
    return name, IMU_MOUNTS[name]


def reading_near_end(pos_dd) -> np.ndarray:
    """Per servo: the measured ``pos_dd`` within ``READING_WATCH_DD`` of a track end, or past it."""
    d = np.asarray(pos_dd, dtype=float)
    return (d <= TRACK_DD[0] + READING_WATCH_DD) | (d >= TRACK_DD[1] - READING_WATCH_DD)


def policy_act(pol: NumpyPolicy):
    """``act(obs, k)``: the policy on its observation, stacked with its meta's history (cleared at k 0)."""
    hist = int(pol.meta.get("history", 0) or 0)
    if not hist:
        return lambda obs, k: pol(obs)
    stack = HistoryStacker(hist)

    def act(obs, k):
        if k == 0:
            stack.reset()
        return pol(stack(obs))
    return act


def policy_decoder(pol: NumpyPolicy, contract: Contract) -> ActionDecoder:
    return ActionDecoder(contract, pol.scale, band_deg=policy_band_deg(pol.meta), bounds=policy_bounds(pol.meta),
                         stand_offset_abd_deg=policy_stand_offset_deg(pol.meta), filter_alpha=policy_filter_alpha(pol.meta))


@dataclass
class GetupSpec:
    act: object                # act(obs, k), k = steps since this get-up began
    decoder: ActionDecoder
    max_getups: int
    timeout_s: float
    phase_s: float             # the time the phase clock takes to reach its end
    phase_restart_s: float     # the time it stays there before a retry; 0: never
    handback_tilt_deg: float = HANDBACK_TILT_DEG
    handback_gyro: float = HANDBACK_GYRO
    handback_rms_deg: float = HANDBACK_RMS_DEG


class Runner:
    def __init__(self, link, contract: Contract, mount: np.ndarray, *, stdin=None, out=print):
        self.link, self.c, self.mount = link, contract, mount
        self.stdin = stdin
        self.out = out
        self.stand_dd = np.asarray(contract.stand_dd, dtype=int)
        self.armed = False
        self.first_seq: int | None = None                 # the arming frame's seq (``check_flags``)
        self.clear = False                                # the first arming frame also clears a latched fault
        self.last_dd = None
        self.steps: dict[str, list] = {k: [] for k in (
            "tick", "t_rx", "t_send", "seq", "obs", "raw", "rel", "dd", "tilt_deg", "late", "heading_psi", "heading_integ",
            "heading_err", "wz_cmd", "cmd", "mode", "tilt_cf_deg", "gyro_norm", "getup_echo", "cur_max_a",
            "reading_flagged")}
        self.stop_reason = ""
        self.warnings: list[str] = []
        self.gyro_bias = np.zeros(3)
        self.robot_offsets = None
        self.command_initial = None
        self.mode = MODE_WALK
        self.fall = FallDetector()
        self.fall_pending = False                         # a fall seen, the get-up not yet entered
        self.stand_phase = False                          # the stand ramp or hold: a fall there raises FallDetected
        self.gf: GravityFilter | None = None
        self.tilt_cf = float("nan")
        self.gyro_norm = 0.0
        self._gf_tick = None
        self._rec_i = 0
        self.getups: list[dict] = []
        self.reading_flags: list[dict] = []
        self._rw_seen: dict[int, dict] = {}
        self._rw_tick0 = None

    # ---- checks ---------------------------------------------------------------------------------------------------
    def preflight(self, clear: bool = False) -> None:
        st = self.link.wait_state(1.0)
        preflight(self.link, clear)
        self.clear = clear
        present(self.link)
        if not st.flags & p.ST_IMU_OK:
            raise SystemExit("IMU absent: power-cycle the board and check ping before a policy run")
        for i in range(1, 13):
            for name, value in {**IDENTIFIED, "ki_position": 0.0, "kp_current": 0.0, "pwm_scale": PWM_SCALE}.items():
                self.link.set_param(i, name, value)
        for i in range(1, 13):
            for name, want in IDENTIFIED.items():
                got = self.link.get_param(i, name)
                if abs(got - want) > 1e-6 + 1e-3 * abs(want):
                    raise SystemExit(f"servo {i} {name} = {got:g}, not {want:g} after writing it")
            got = self.link.get_param(i, "pwm_scale")
            if not abs(got - PWM_SCALE) <= PWM_SCALE_TOL * PWM_SCALE:
                raise SystemExit(f"servo {i} pwm_scale = {got:g}, not {PWM_SCALE:g} (+-{PWM_SCALE_TOL:.0%}) after writing it")
        offs, stored = read_offsets(self.link)
        self.robot_offsets = offs
        diff = np.abs(offs - self.c.offsets_deg)
        if not all(stored) or np.any(diff > OFFSET_REFUSE_DEG):
            raise SystemExit(f"the robot's calibration offsets {np.round(offs, 2).tolist()} (stored {list(map(bool, stored))}) "
                             f"differ from the contract's {np.round(self.c.offsets_deg, 2).tolist()} by up to "
                             f"{diff.max():.1f} deg: deploy the contract made from this calibration")
        if np.any(diff > 1e-3):
            msg = (f"WARNING: the robot's calibration offsets {np.round(offs, 2).tolist()} differ from the contract's "
                   f"{np.round(self.c.offsets_deg, 2).tolist()}; the policy runs on the contract's stand")
            self.warnings.append(msg)
            self.out(msg)

    def measure_gyro_bias(self, seconds: float = 1.0) -> np.ndarray:
        """The gyro's mean over ``seconds`` with the servos idle (sensor frame, rad/s); refuses one over
        ``GYRO_BIAS_MAX`` on any axis."""
        end = time.monotonic() + seconds
        st, g = None, []
        while time.monotonic() < end:
            st = self.link.wait_state(0.1, after=st)
            if imu_valid(st, p.ST_IMU_OK):
                g.append(np.asarray(st.gyr_ddps, dtype=float) * 0.1 * DEG)
        if len(g) < 0.5 * seconds / 0.005:
            raise SystemExit("too few valid IMU frames to measure the gyro bias")
        bias = np.mean(g, axis=0)
        if np.abs(bias).max() > GYRO_BIAS_MAX:
            raise SystemExit(f"the gyro read {np.round(bias, 4).tolist()} rad/s while the robot should be still (at most "
                             f"{GYRO_BIAS_MAX:g} on any axis): keep the robot still for the first second and start again")
        self.gyro_bias = bias
        return self.gyro_bias

    # ---- sending --------------------------------------------------------------------------------------------------
    def send(self, dd) -> int:
        dd = np.asarray(dd)
        if not np.all(np.isfinite(dd)):
            raise StopRun("non-finite target")
        dd = np.clip(np.rint(dd), 0, 3100).astype(int)
        mode = p.SA_GETUP if self.mode == MODE_GETUP else 0
        flags = 0 if self.armed else p.SA_ARM | (p.SA_CLEAR if self.clear else 0)
        seq = self.link.set_all(dd, enable=0x0FFF, flags=flags | mode, cur_cap_ma=CAP_MA)
        if not self.armed:
            self.first_seq = seq
        self.armed = True
        self.last_dd = dd
        return seq

    def watch(self) -> None:
        """After each send: the link, the firmware's state since our ARM (not armed: it refused the ARM or disarmed), IMU,
        Enter."""
        try:
            check_flags(self.link, self)
        except Exception as exc:
            raise Disarm(f"link: {exc}") from exc
        st = self.link.latest
        if not imu_valid(st, p.ST_IMU_OK):
            raise StopRun("IMU frame invalid")
        self.feed_imu()
        if self.enter_pressed():
            raise StopRun("Enter")

    def enter_pressed(self) -> bool:
        if self.stdin is None:
            return False
        r, _, _ = select.select([self.stdin], [], [], 0)
        if r:
            self.stdin.readline()
            return True
        return False

    # ---- the complementary filter and the fall detector ------------------------------------------------------------
    def start_imu(self) -> None:
        """Start the complementary filter from the next frame (the gyro's bias known)."""
        self.gf = GravityFilter()
        self._gf_tick = None
        self._rec_i = len(self.link.record)

    def imu_update(self, s) -> None:
        """One STATE frame (each tick once, in order; invalid IMU frames skipped)."""
        if self.gf is None or not imu_valid(s, p.ST_IMU_OK):
            return
        tick = int(s.tick)
        if self._gf_tick is not None:
            gap = (tick - self._gf_tick) & 0xFFFFFFFF
            if gap == 0 or gap >= 0x80000000:
                return                                    # already fed, or older
            dt = min(gap, 20) * 0.005
        else:
            dt = 0.005
        self._gf_tick = tick
        gyro = self.mount @ (np.asarray(s.gyr_ddps, dtype=float) * 0.1 * DEG - self.gyro_bias)
        self.tilt_cf = self.gf.update(gyro, self.mount @ np.asarray(s.acc_mg, dtype=float), dt)
        self.gyro_norm = float(np.linalg.norm(gyro))
        if self.mode == MODE_WALK:
            self.fall.update(self.tilt_cf, self.gyro_norm, tick * 0.005)
            if self.fall.fallen:
                self.fall_pending = True

    def feed_imu(self) -> None:
        """Every recorded frame not yet fed."""
        if self.gf is None:
            return
        rec = self.link.record
        for s in rec[self._rec_i:]:
            self.imu_update(s)
        self._rec_i = len(rec)
        if self.fall_pending and self.stand_phase:
            raise FallDetected(self.fall.trigger)

    # ---- the reading watch (the run's start and get-ups; it prints and logs, nothing stops) --------------------------
    def reading_watch_reset(self) -> None:
        self._rw_seen = {}

    def reading_watch(self, s, getup_n: int) -> None:
        """A servo reading near a track end (filtered, ``feedback.position``): printed and flagged once per episode. Frames
        without fresh feedback from every board are skipped: their positions can be a boot's zeros."""
        if s.boards != 0x0F:
            return
        tick = int(s.tick)
        if self._rw_tick0 is None:
            self._rw_tick0 = tick
        d = position(s)
        near = reading_near_end(d)
        if not near.any():
            return
        names = self.c.meta.get("servo_order") or [str(i + 1) for i in range(12)]
        for i in np.flatnonzero(near).tolist():
            low = d[i] < 0.5 * (TRACK_DD[0] + TRACK_DD[1])
            ev = self._rw_seen.get(i)
            if ev is not None:
                ev["extreme_dd"] = min(ev["extreme_dd"], float(d[i])) if low else max(ev["extreme_dd"], float(d[i]))
                continue
            t_s = ((tick - self._rw_tick0) & 0xFFFFFFFF) * 0.005
            ev = {"servo": i + 1, "name": names[i], "dd": float(d[i]), "track_end": TRACK_DD[0] if low else TRACK_DD[1],
                  "getup": getup_n, "tick": tick, "t_s": t_s, "extreme_dd": float(d[i])}
            self._rw_seen[i] = ev
            self.reading_flags.append(ev)
            where = f"in GETUP {getup_n}" if getup_n else "at the start"
            self.out(f"ADVISORY (log only): servo {i + 1} ({names[i]}) reads {d[i]:.0f} dd, within "
                     f"{READING_WATCH_DEG:g} deg of the track end {ev['track_end']:.0f}, {where}, t {t_s:.2f} s")

    def reading_watch_start(self) -> None:
        """The first frame with fresh feedback from every board; the preflight, next, reports a link that has none."""
        try:
            for _ in range(20):
                st = self.link.wait_state(1.0)
                if st.boards == 0x0F:
                    self.reading_watch(st, 0)
                    return
        except Exception:
            return

    # ---- the first step -----------------------------------------------------------------------------------------------
    def warm_up(self, ob: ObservationBuilder, heading: HeadingLoop | None, walker_act, walker_decoder: ActionDecoder,
                getup: GetupSpec) -> None:
        """One step of each policy, nothing sent. A first call can hold the interpreter for tens of ms with every thread
        stalled, the serial reader's too: on numpy 2, the velocity filter's first ``np.median`` imports ``numpy.ma``."""
        st = self.link.wait_state(0.5)
        ob.push(st)
        if heading is not None:
            heading.push(ob.gyro_body(st), ob.gravity_body(st))
            heading.wz()
        for act, dec, command in ((walker_act, walker_decoder, (0.0, 0.0, 0.0)),
                                  (getup.act, getup.decoder, getup_phase(0.0, getup.phase_s))):
            obs = ob.observe(command, dec.prev_action, dec.obs_offset)
            target_refusal(dec, dec(np.asarray(act(obs, 0), dtype=float)))
            dec.reset()
        tilt_deg(st.acc_mg)
        ob.reset()
        if heading is not None:
            heading.reset()

    def keep_fed(self) -> None:
        """The last target again, so that the firmware keeps the servos armed while the first step waits for its frames."""
        if self.armed:
            self.send(self.last_dd)

    # ---- moving to the stand --------------------------------------------------------------------------------------
    def ramp(self, to_dd, seconds: float, frm=None, stoppable: bool = True) -> None:
        frm = np.asarray(self.last_dd if frm is None else frm, dtype=float)
        to = np.asarray(to_dd, dtype=float)
        pacer = Pacer(CONTROL_DT)
        n = max(1, int(round(seconds / CONTROL_DT)))
        for k in range(n):
            self.send(frm + (to - frm) * smoothstep((k + 1) / n))
            pacer.wait()
            try:
                self.watch()
            except StopRun:
                if stoppable:
                    raise

    def hold(self, dd, seconds: float, stoppable: bool = True) -> None:
        pacer = Pacer(CONTROL_DT)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(dd)
            pacer.wait()
            try:
                self.watch()
            except StopRun:
                if stoppable:
                    raise

    def safe_end(self) -> None:
        """Ramp to the stand, hold, disarm."""
        try:
            if self.armed and self.last_dd is not None:
                self.ramp(self.stand_dd, 1.0, stoppable=False)
                self.hold(self.stand_dd, HOLD_AFTER_S, stoppable=False)
        except Disarm as exc:
            self.stop_reason += f"; {exc}"
        finally:
            self.link.disarm()
            self.armed = False

    # ---- the loop -------------------------------------------------------------------------------------------------
    def loop(self, walker_act, command, seconds: float, ob: ObservationBuilder, walker_decoder: ActionDecoder,
             heading: HeadingLoop | None, getup: GetupSpec, start_getup: bool = False, start_trigger: str | None = None) -> None:
        """Policy steps locked to the STATE ticks. ``command``: (vx, vy, wz). ``start_getup``: a fall already happened, the
        first step is in the get-up (``start_trigger`` names it when the robot was placed lying down)."""
        link = self.link
        rec = link.record
        idx = len(rec)
        self.keep_fed()
        st = link.wait_state(0.05)
        ob.reset()
        walker_decoder.reset()
        ob.push(st)
        if heading is not None:
            heading.reset()
        command = tuple(float(v) for v in command)
        self.command_initial = list(command)
        next_tick = int(st.tick) + TICKS_PER_STEP
        late_run = 0
        n_steps = int(round(seconds / CONTROL_DT))
        k_mode = 0                            # steps since the current mode began
        if start_getup:
            self.fall_pending = True
        g_entry = g_echo = g_phase = 0        # the get-up's entry tick, last echo tick, phase clock's origin tick
        hb_run = 0                            # consecutive steps meeting the hand-back criterion
        for k in range(n_steps):
            while True:                       # every frame up to the step's tick, in order
                try:
                    st = link.wait_state(0.025, after=st)
                except Exception as exc:
                    raise StopRun(f"STATE gap: {exc}") from exc
                new = rec[idx:]
                idx = len(rec)
                for s in new:
                    if ob.last_tick is None or ((int(s.tick) - ob.last_tick) & 0xFFFFFFFF) < 1000:
                        if int(s.tick) != ob.last_tick:
                            ob.push(s)
                            if heading is not None and imu_valid(s, p.ST_IMU_OK):
                                heading.push(ob.gyro_body(s), ob.gravity_body(s))
                    if not imu_valid(s, p.ST_IMU_OK):
                        raise StopRun("IMU frame invalid")
                    self.imu_update(s)
                    if self.mode == MODE_GETUP:
                        if s.flags & p.ST_GETUP:
                            g_echo = int(s.tick)
                        self.reading_watch(s, len(self.getups))
                if ((int(st.tick) - next_tick) & 0xFFFFFFFF) < 0x80000000:
                    break
                if k == 0:
                    self.keep_fed()
            late = (int(st.tick) - next_tick) & 0xFFFFFFFF
            late_run = late_run + 1 if late > 0 else 0
            if late_run >= LATE_STEPS_STOP:
                raise StopRun("two late steps in a row")
            next_tick = int(st.tick) + TICKS_PER_STEP
            tcf, gn = self.tilt_cf, self.gyro_norm
            pos = self.c.rel_from_dd(position(st))
            if self.mode == MODE_WALK and self.fall_pending:                  # walk -> get-up
                g_now = self.gf.g if self.gf is not None else None
                ev = {"n": len(self.getups) + 1, "step": k, "tick": int(st.tick), "t": time.monotonic(),
                      "trigger": (start_trigger or "stand ramp/hold: " + str(self.fall.trigger))
                      if start_getup and k == 0 else str(self.fall.trigger), "tilt_cf_deg": self.tilt_cf, "which": "getup",
                      "roll_deg": None if g_now is None else math.degrees(math.atan2(-g_now[1], -g_now[2])),
                      "gravity": None if g_now is None else [float(v) for v in g_now]}
                if len(self.getups) >= getup.max_getups:
                    raise Disarm(f"fall {ev['n']} ({ev['trigger']}): --getup-max {getup.max_getups} reached")
                self.getups.append(ev)
                self.out(f"fall ({ev['trigger']}, tilt {self.tilt_cf:.0f} deg): GETUP {ev['n']}")
                self.mode, self.fall_pending = MODE_GETUP, False
                ob.reset()
                ob.push(st)
                getup.decoder.reset(pos)
                g_entry = g_echo = g_phase = int(st.tick)
                if not (start_trigger and k == 0):         # a placed start's reading watch goes on from the run's start
                    self.reading_watch_reset()
                self.reading_watch(st, ev["n"])
                hb_run, k_mode = 0, 0
            elif self.mode == MODE_GETUP:
                ev = self.getups[-1]
                if ((int(st.tick) - g_echo) & 0xFFFFFFFF) * 0.005 > ECHO_TIMEOUT_S:
                    msg = "the firmware did not enter get-up mode: it doesn't support the get-up"
                    self.warnings.append(msg)
                    self.out("WARNING: " + msg)
                    ev.update(end="no get-up mode", end_step=k)
                    raise Disarm("get-up: " + msg)
                ok = (tcf < getup.handback_tilt_deg and gn < getup.handback_gyro
                      and math.degrees(float(np.sqrt(np.mean(pos ** 2)))) < getup.handback_rms_deg)
                hb_run = hb_run + 1 if ok else 0
                t_in = ((int(st.tick) - g_entry) & 0xFFFFFFFF) * 0.005
                if hb_run >= HANDBACK_STEPS:                                  # get-up -> walk
                    ev.update(end="hand-back", end_step=k, end_tick=int(st.tick), seconds=t_in)
                    self.out(f"hand-back after {t_in:.2f} s in GETUP {ev['n']}")
                    self.mode = MODE_WALK
                    self.reading_watch_reset()
                    self.fall.reset()
                    ob.reset()
                    ob.push(st)
                    walker_decoder.reset(pos)
                    if heading is not None:
                        heading.relatch()
                    k_mode = 0
                elif t_in > getup.timeout_s:
                    ev.update(end="timeout", end_step=k, seconds=t_in)
                    raise Disarm(f"get-up: not up after {getup.timeout_s:g} s (limp)")
                elif getup.phase_restart_s > 0:
                    t_ph = ((int(st.tick) - g_phase) & 0xFFFFFFFF) * 0.005
                    if t_ph >= getup.phase_s + getup.phase_restart_s - 1e-9 and tcf > PHASE_RESTART_TILT_DEG:
                        g_phase = int(st.tick)
                        ev.setdefault("phase_restarts", []).append(k)
                        self.out(f"GETUP {ev['n']}: phase restarted after {t_in:.2f} s (tilt {tcf:.0f} deg)")
            cmd = command
            walking = self.mode == MODE_WALK
            if heading is not None:
                if not walking:               # during a get-up the heading is tracked, the loop idle
                    heading.err = 0.0
                    cmd = (command[0], command[1], 0.0)
                elif abs(command[2]) < 1e-6:
                    cmd = (command[0], command[1], command[2] + heading.wz())
                else:                         # a commanded turn is sent alone
                    heading.err = 0.0
                self.steps["heading_psi"].append(heading.psi)
                self.steps["heading_integ"].append(heading.integ)
                self.steps["heading_err"].append(heading.err)
                self.steps["wz_cmd"].append(cmd[2])
            self.steps["cmd"].append((*command, cmd[2]))
            if walking:
                obs = ob.observe(cmd, walker_decoder.prev_action, walker_decoder.obs_offset)
                raw = np.asarray(walker_act(obs, k_mode), dtype=float)
                dec = walker_decoder
            else:
                phi_t = ((int(st.tick) - g_phase) & 0xFFFFFFFF) * 0.005
                obs = ob.observe(getup_phase(phi_t, getup.phase_s), getup.decoder.prev_action, getup.decoder.obs_offset)
                raw = np.asarray(getup.act(obs, k_mode), dtype=float)
                dec = getup.decoder
            k_mode += 1
            if not np.all(np.isfinite(raw)):
                raise StopRun("non-finite action")
            d = dec(raw)
            why = target_refusal(dec, d)
            if why:
                raise Disarm(f"target check: {why}")
            t_send = time.monotonic()
            seq = self.send(d["dd"])
            if k == 0:
                self.out("policy start")
            self.watch()
            for key, val in (("tick", int(st.tick)), ("t_rx", st.t_rx), ("t_send", t_send), ("seq", seq),
                             ("obs", obs), ("raw", raw), ("rel", d["rel"]), ("dd", d["dd"]),
                             ("tilt_deg", tilt_deg(st.acc_mg)), ("late", late), ("mode", int(not walking)),
                             ("tilt_cf_deg", tcf), ("gyro_norm", gn), ("getup_echo", bool(st.flags & p.ST_GETUP)),
                             ("cur_max_a", float(np.max(np.abs(np.asarray(st.cur_ma, dtype=float)))) / 1000.0),
                             ("reading_flagged", not walking and bool(self._rw_seen))):
                self.steps[key].append(val)

    def save(self, log: str, meta: dict) -> None:
        base = Path(log)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(meta, stop_reason=self.stop_reason, warnings=self.warnings, gyro_bias_rad_s=self.gyro_bias.tolist(),
                    stand=self.stand_dd.tolist(), cap=CAP_MA)
        save_log(self.link, str(base.with_suffix(".npz")), None, meta)
        arrays = {k: np.asarray(v) for k, v in self.steps.items()}
        np.savez_compressed(base.with_suffix(".policy.npz"), meta=json.dumps(meta), **arrays)
        offs = self.robot_offsets if self.robot_offsets is not None else self.c.offsets_deg
        base.with_suffix(".stand.json").write_text(json.dumps(
            {"stand_dd": self.stand_dd.tolist(), "stand_ticks": np.asarray(self.c.stand_ticks).tolist(),
             "offsets_deg": np.asarray(offs, dtype=float).tolist(), "source": "the contract; offsets read from the robot"},
            indent=1))


# ---------------------------------------------------------------------------------------------------- commands
def layout_refusal(pol: NumpyPolicy) -> str | None:
    """Why the runner cannot feed this policy, or None: it observes more than the standard terms."""
    if has_heading_obs(pol.meta):
        return "this policy observes the heading error: the runner does not feed that input"
    want = obs_size() * max(int(pol.meta.get("history", 0) or 0), 1)
    have = int(pol.layers[0][0].shape[1])
    return None if have == want else f"the policy has {have} inputs, its meta says {want}"


def policy_calibration(folder, meta: dict) -> tuple[Contract | None, Path | None, str | None]:
    """The calibration a policy folder carries (``checks.folder_calibration``) checked against the policy's meta
    (``checks.calibration_file_refusal``): (contract, its file, None), or (None, file or None, why it is refused)."""
    path, why = folder_calibration(folder)
    if why:
        return None, None, why
    why = calibration_file_refusal(meta, path)
    if why:
        return None, path, why
    return Contract.load(path), path, None


def cmd_check(a) -> int:
    """No port: the policy under this numpy against its export's fixtures, the calibration its folder carries, the
    robot it was trained for, its layout, its band and bounds, the decoder self-test."""
    folder = Path(a.policy)
    path = folder / "policy.npz"
    pol = NumpyPolicy(path)
    d = np.load(path)
    err = float(np.abs(pol(d["fixture_obs"]) - d["fixture_act"]).max())
    tol = fixture_tolerance(d["fixture_act"])
    c, cal, cal_why = policy_calibration(folder, pol.meta)
    layout_why = layout_refusal(pol)
    robot_why = robot_refusal(pol.meta)
    x = d["fixture_obs"][0]
    t0 = time.perf_counter()
    for _ in range(1000):
        pol(x)
    us = (time.perf_counter() - t0) * 1e3
    print(f"fixtures: max |numpy - torch| {err:.2e} ({'ok' if err < tol else 'FAIL'}, limit {tol:.2e})")
    if cal_why:
        print(f"calibration: FAIL: {cal_why}")
    else:
        have = paths.sha256(cal)
        same = pol.meta.get("contract_sha256") == have
        print(f"calibration: {'ok' if same else 'ok (exported for an earlier calibration)'} ({cal.name} {have[:12]})")
    print(f"robot: {'ok' if not robot_why else 'FAIL: ' + robot_why} (servo model {pol.meta.get('actuator')}, legs "
          f"{(pol.meta.get('legs') or {}).get('legs')})")
    print(f"inputs: {int(pol.layers[0][0].shape[1])} ({'ok' if not layout_why else 'FAIL: ' + layout_why})")
    print(f"inference: {us:.1f} us per step; numpy {np.__version__}")
    band_why = "not run (no calibration)" if c is None else band_refusal(pol.meta, c)
    print(f"kind {policy_kind(pol.meta)}, band deg {policy_band_deg(pol.meta).tolist()} "
          f"({'FAIL: ' + band_why if band_why else 'ok'})")
    test_why = "not run (the band check failed)"
    if not band_why:
        b = policy_bounds(pol.meta)
        print(f"bounds: {'none' if b is None else b.mode}, stand offset {policy_stand_offset_deg(pol.meta):g} deg outward, "
              f"target filter alpha {policy_filter_alpha(pol.meta):g}")
        test_why = decoder_selftest(ActionDecoder(c, d["scale"], band_deg=policy_band_deg(pol.meta), bounds=b,
                                                  stand_offset_abd_deg=policy_stand_offset_deg(pol.meta),
                                                  filter_alpha=policy_filter_alpha(pol.meta)), c)
    print(f"decoder self-test (targets inside the bounds, {TARGET_END_DEG:g} deg from the track ends): "
          f"{'FAIL: ' + test_why if test_why else 'ok'}")
    return 0 if err < tol and not cal_why and not robot_why and not layout_why and not band_why and not test_why else 1


def cmd_imu(link, a) -> int:
    """The servos stay idle. Prints the tilt and the gravity direction in each mounting candidate, to tell which one fits:
    nose down, gravity's x is positive; right side down, its y is negative."""
    mounts = dict(IMU_MOUNTS)
    name, R = load_mount()
    if name == "custom":
        mounts["custom"] = R
    link.record = []
    end = time.monotonic() + a.seconds
    st, last_print = None, 0.0
    while time.monotonic() < end:
        st = link.wait_state(0.5, after=st)
        if time.monotonic() - last_print > 0.2:
            last_print = time.monotonic()
            acc = np.asarray(st.acc_mg, dtype=float)
            cands = "  ".join(f"{k}: g=({', '.join(f'{v:+.2f}' for v in -(R @ acc) / np.linalg.norm(acc))})"
                              for k, R in mounts.items()) if np.linalg.norm(acc) > 0 else "acc 0"
            print(f"tilt {tilt_deg(acc):5.1f}  imu {'ok' if st.flags & p.ST_IMU_OK else 'ABSENT'}  "
                  f"gyro dps {np.round(np.asarray(st.gyr_ddps) * 0.1, 1).tolist()}  {cands}", flush=True)
    if a.log:
        save_log(link, a.log, None, {"cmd": "imu", "seconds": a.seconds})
    return 0


def cmd_run(link, a) -> int:
    mount_name, mount = load_mount()
    walker = NumpyPolicy(Path(a.policy) / "policy.npz")
    gpol = NumpyPolicy(Path(a.getup_policy) / "policy.npz")
    why, contract, cals = None, None, {}
    if policy_kind(walker.meta) != "walk":
        why = f"--policy is a {policy_kind(walker.meta)!r} policy, not a walking policy"
    elif policy_kind(gpol.meta) != "getup":
        why = f"--getup-policy is a {policy_kind(gpol.meta)!r} policy, not a get-up policy"
    elif not walker.meta.get("omni"):
        why = "--policy does not observe the whole command (vx, vy, wz)"
    elif not gpol.meta.get("getup_phase_s"):
        why = "--getup-policy has no phase clock (meta getup_phase_s)"
    else:
        for flag, folder, pol in (("--policy", a.policy, walker), ("--getup-policy", a.getup_policy, gpol)):
            c, cals[flag], why = policy_calibration(folder, pol.meta)
            if why:
                why = f"{flag}: {why}"
                break
            contract = contract or c
        why = why or pair_calibration_refusal(cals["--policy"], cals["--getup-policy"])
    if not why:
        for flag, pol in (("--policy", walker), ("--getup-policy", gpol)):
            why = robot_refusal(pol.meta) or band_refusal(pol.meta, contract) or layout_refusal(pol)
            if why:
                why = f"{flag}: {why}"
                break
    if why:
        print(f"refused: {why}", flush=True)
        return 2
    decoder = policy_decoder(walker, contract)
    getup = GetupSpec(act=policy_act(gpol), decoder=policy_decoder(gpol, contract), max_getups=int(a.getup_max),
                      timeout_s=float(a.getup_timeout), phase_s=float(gpol.meta["getup_phase_s"]),
                      phase_restart_s=float(a.getup_phase_restart), handback_tilt_deg=float(a.handback_tilt),
                      handback_gyro=float(a.handback_gyro), handback_rms_deg=float(a.handback_rms))
    for flag, dec in (("--policy", decoder), ("--getup-policy", getup.decoder)):
        why = decoder_selftest(dec, contract)
        if why:
            print(f"refused: {flag}: decoder self-test: {why}", flush=True)
            return 2
    heading = HeadingLoop() if a.heading_loop else None
    runner = Runner(link, contract, mount, stdin=sys.stdin if sys.stdin.isatty() else None)
    runner.reading_watch_start()              # before the preflight, whose position check refuses these readings
    runner.preflight(a.clear)
    link.record = []
    runner.measure_gyro_bias(1.0)
    ob = ObservationBuilder(contract, mount, runner.gyro_bias)
    runner.start_imu()
    walk_stand = decoder.stand_dd()
    walker_act = policy_act(walker)
    meta = {"cmd": "run", "start_mode": a.start_mode, "seconds": a.seconds, "vx": a.vx, "vy": a.vy, "wz": a.wz,
            "policy": str(Path(a.policy).resolve()), "policy_sha256": paths.sha256(Path(a.policy) / "policy.npz"),
            "getup_policy": str(Path(a.getup_policy).resolve()),
            "getup_policy_sha256": paths.sha256(Path(a.getup_policy) / "policy.npz"),
            "contract_sha256": paths.sha256(cals["--policy"]), "calibration_file": cals["--policy"].name,
            "mount": mount_name, "mount_R": mount.tolist(), "mount_file_sha256": paths.sha256(IMU_MOUNT_JSON), "stand_dd_policy": [int(v) for v in walk_stand],
            "heading_loop": None if heading is None else heading.describe(), "getup_max": getup.max_getups,
            "getup_timeout_s": getup.timeout_s, "getup_phase_s": getup.phase_s, "getup_phase_restart_s": getup.phase_restart_s,
            "target_end_deg": TARGET_END_DEG,
            "getup_rules": {"cf_tau_s": CF_TAU_S, "cf_gate_g": list(CF_GATE_G),
                            "fall": [FALL_TILT_DEG, FALL_HOLD_S, FALL_FAST_TILT_DEG, FALL_FAST_GYRO],
                            "handback": [getup.handback_tilt_deg, getup.handback_gyro, getup.handback_rms_deg, HANDBACK_STEPS],
                            "restart_tilt_deg": PHASE_RESTART_TILT_DEG, "echo_timeout_s": ECHO_TIMEOUT_S},
            "reading_watch": {"deg": READING_WATCH_DEG}}
    runner.warm_up(ob, heading, walker_act, decoder, getup)
    # A full collection over the log's objects can stall every thread for tens of ms: none while armed.
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        start = present(link)
        fell = False
        if a.start_mode == "getup":
            fell = True
            tilt0 = tilt_deg(link.latest.acc_mg)
            meta["start_tilt_deg"] = tilt0
            if tilt0 < FALL_FAST_TILT_DEG:
                msg = f"--start-mode getup with the robot at {tilt0:.0f} deg tilt (not lying down): the get-up starts anyway"
                runner.warnings.append(msg)
                print("WARNING: " + msg, flush=True)
        else:
            runner.stand_phase = True
            try:
                runner.ramp(walk_stand, a.stand_ramp, frm=start)
                runner.hold(walk_stand, a.hold_before)
            except FallDetected:
                fell = True
            finally:
                runner.stand_phase = False
        runner.loop(walker_act, (a.vx, a.vy, a.wz), a.seconds, ob, decoder, heading, getup, start_getup=fell,
                    start_trigger="start mode getup" if a.start_mode == "getup" else None)
        runner.stop_reason = "completed"
        if runner.mode == MODE_GETUP:
            runner.stop_reason = "completed in GETUP (limp)"
            link.disarm()
        else:
            runner.safe_end()
    except StopRun as exc:
        runner.stop_reason = f"stop: {exc}"
        print(runner.stop_reason, flush=True)
        if runner.mode == MODE_GETUP:
            runner.stop_reason += " (in GETUP: limp)"
            link.disarm()
        else:
            runner.safe_end()
    except (Disarm, KeyboardInterrupt) as exc:
        runner.stop_reason = f"disarm: {str(exc) or 'Ctrl-C'}"
        link.disarm()
        print(runner.stop_reason, flush=True)
    except BaseException:
        link.disarm()
        raise
    finally:
        gc.enable()
        if a.log:
            meta.update(missing_ticks=ob.missing, command_initial=runner.command_initial,
                        reading_flags=runner.reading_flags, getups=runner.getups)
            runner.save(a.log, meta)
    lat = np.asarray(link.latency, dtype=float) * 1000
    if len(lat):
        print(f"SET_ALL -> STATE echo ms: p50 {np.percentile(lat, 50):.2f} p95 {np.percentile(lat, 95):.2f}; "
              f"missing ticks {ob.missing}; steps {len(runner.steps['tick'])}; {runner.stop_reason}")
    if getattr(link, "feedback", None) is not None:
        print(link.feedback.report(), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mp4_rl.deploy.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyAMA0")
    ap.add_argument("--baud", type=int, default=921_600)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="no port: check a policy against its calibration and this numpy")
    s.add_argument("--policy", default=str(paths.WALKER_POLICIES / "sku2"))
    s = sub.add_parser("imu", help="servos idle: print the tilt and gravity in each IMU mounting candidate")
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--log")
    s = sub.add_parser("run", help="the two policies, closed loop")
    s.add_argument("--policy", default=str(paths.WALKER_POLICIES / "sku2"), help="the walking policy's directory")
    s.add_argument("--getup-policy", default=str(paths.GETUP_POLICIES / "sku2"), help="the get-up policy's directory")
    s.add_argument("--vx", type=float, default=0.0, help="m/s, + = forward")
    s.add_argument("--vy", type=float, default=0.0, help="m/s, + = left")
    s.add_argument("--wz", type=float, default=0.0, help="rad/s, + = turn left")
    s.add_argument("--seconds", type=float, default=10.0, help="the policy loop's length, get-ups included")
    s.add_argument("--heading-loop", action=argparse.BooleanOptionalAction, default=True,
                   help="hold the heading with the gyro while no turn is commanded")
    s.add_argument("--stand-ramp", type=float, default=1.0, help="s of the ramp to the stand (default %(default)s)")
    s.add_argument("--hold-before", type=float, default=0.0, help="s at the stand before the policy (default %(default)s)")
    s.add_argument("--start-mode", choices=("walk", "getup"), default="walk",
                   help="getup: the robot is placed lying down; no ramp, the loop starts in the get-up")
    s.add_argument("--getup-max", type=int, default=10, help="get-ups per run; the next fall goes limp (default %(default)s)")
    s.add_argument("--getup-timeout", type=float, default=20.0,
                   help="s in a get-up before giving up and going limp (default %(default)s)")
    s.add_argument("--getup-phase-restart", type=float, default=1.0, metavar="S",
                   help="the retry: restart the get-up's phase clock S s after it reached its end, if the robot is still "
                        "tilted over 45 deg; 0: never (default %(default)s)")
    s.add_argument("--handback-tilt", type=float, default=HANDBACK_TILT_DEG, metavar="DEG",
                   help="hand back to the walker below this filtered tilt (default %(default)s)")
    s.add_argument("--handback-rms", type=float, default=HANDBACK_RMS_DEG, metavar="DEG",
                   help="... and with the pose within this rms of the stand (default %(default)s)")
    s.add_argument("--handback-gyro", type=float, default=HANDBACK_GYRO, metavar="RAD_S",
                   help="... and below this rate (default %(default)s)")
    s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
    s.add_argument("--log", help="log base path, without suffix")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    from mp4_rl.deploy.mp4link.link import Link
    link = Link(a.port, a.baud)
    try:
        return cmd_imu(link, a) if a.cmd == "imu" else cmd_run(link, a)
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
