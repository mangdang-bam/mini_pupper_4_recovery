"""The classic controller: the official firmware's Stanford trot, run in simulation as a baseline walker.

Source. ``data/classic_gait/stanford_gait.{h,cc}`` and ``stanford_kinematics.{h,cc}`` are the official firmware's files,
unchanged (``SOURCES``: the official firmware repository minipupper_4_mcu, branch firmware-dev, d272000, ``main/robot/``).
``classic_host.c`` includes the gait's source so that many robots each run the firmware's own
``stanford_gait_step`` with their own state, and applies the firmware's servo write. Built with gcc
(``-ffp-contract=off``) into ``~/.cache/mp4_recovery_classic/`` keyed by the sources' hash.

What the firmware runs in its walk (``robot.cc`` GaitCmd::StanfordWalk, the Pi path): every 15 ms (``TICK_MS``)
``stanford_gait_step(vx, vy, wz, s_cfg.height, SG_NATIVE_CLEARANCE_MM)`` then ``sg_write``:
``set_servo_angle(i, sdeg[i] + s_offset[i])``, ``sig = 511 + (int)(deg * 10240 / scale_dd)``. The Pi's velocity
(``set_velocity``) is clamped to +-200 mm/s forward and sideways, +-2.0 rad/s yaw (``firmware_command``); a velocity above
2 mm/s or 0.02 rad/s starts the walk from the stand (``stanford_gait_reset``), and a zero command after that steps in
place. Defaults: ``s_cfg.height`` 70 mm (``Config``, robot.h; SK_NEUTRAL_HEIGHT_MM 70 is the IK's stand; the BSP's 80 mm,
SG_NATIVE_HEIGHT_MM, is not what the walk passes), clearance SG_NATIVE_CLEARANCE_MM = 20 mm.

Calibration mapping. The firmware's IK returns each servo's degrees relative to its neutral stance (feet under the hips,
``height`` below: 0 at the stand), and adds the per-servo offset before the tick formula. Our contract's ``servo_deg`` is
the same quantity (``contract.py``: ``tick = 511 + (int)((servo_deg + offset) * 10240 / 3100)``, ``offsets_deg`` the
robot's calibration, ``stand_ticks`` = the ticks of servo_deg 0), its ``rel = sign * radians(servo_deg)`` and the
model's stand (``stand_joint_rad``) is that neutral pose. So the firmware's tick for servo i, with the contract's offsets
and a 3100 dd range, is the tick our decoder sends for the policy target ``contract.rel_from_ticks(tick)``; at the
neutral feet every tick is the contract's ``stand_ticks`` (``tests/cpu/test_classic_gait.py``). The one formula
difference: firmware-dev multiplies by ``10240 / scale_dd`` (computed first), the contract and the earlier firmware divide
``deg * 10240`` by 3100; the two can differ by one tick at a boundary (``ch_servo``'s ``mul_first``; counted in the test).

In simulation (``ClassicDriver``): each env's gait runs on the 1 ms physics clock, a tick at every 15 ms from the walk's
start, its targets applied at the substep the firmware would write them (no 50 Hz resampling), each tick's dd the
firmware's (``rel_from_ticks`` then the decoder's quantisation, the same arithmetic as a policy's target). The policy
decoder's walk caps (band, step limit, ``limits.decoder_bounds("walk")``) are not applied (the firmware has none):
``ClassicDriver`` records by how much the targets exceed them; ``clamp=True`` applies the band and the bounds.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from mp4_recovery.contract import BAND_DEG, STEP_LIMIT_DEG, Contract

HERE = Path(__file__).resolve().parent / "data" / "classic_gait"
FIRMWARE_FILES = ("stanford_gait.h", "stanford_gait.cc", "stanford_kinematics.h", "stanford_kinematics.cc")
HOST = HERE / "classic_host.c"
SOURCES = {"repository": "minipupper_4_mcu (official), branch firmware-dev", "commit": "d2720006a795360506fb29713c7fc080329fc098",
           "last_change_to_these_files": "a8c93dd21545025baf07f977a3f322ed56faa016", "path": "main/robot/"}
TICK_MS = 15                      # SG_DT
HEIGHT_MM = 70.0                  # Config::height (robot.h), the walk's body height
CLEARANCE_MM = 20.0               # SG_NATIVE_CLEARANCE_MM, what the walk passes
VXY_MAX_MM_S = 200.0              # SG_SPEED_MAX_MM_S, JOY_VY_MAX: the Pi path's clamp
WZ_MAX = 2.0                      # JOY_WZ_MAX
MOVE_MM_S, MOVE_WZ = 2.0, 0.02    # set_velocity: a command above these starts the walk
SCALE_DD = 3100.0                 # the boards' range (contract.py assumes 3100)
CYCLE_TICKS = 24                  # SG_PHASE_LENGTH
STEADY_MS = 720                   # the walk's start transient is over after two cycles (a stance leg slides 18 ticks,
                                  # twice the steady stride, before its first swing): ``steady_*`` statistics after it


def source_hash() -> str:
    return hashlib.sha256(b"".join((HERE / f).read_bytes() for f in FIRMWARE_FILES + ("classic_host.c",))).hexdigest()


def library():
    """The host library (built on first use)."""
    h = source_hash()[:12]
    lib = Path.home() / ".cache/mp4_recovery_classic" / f"libclassic_{h}.so"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-ffp-contract=off", "-x", "c", f"-I{HERE}", str(HOST), str(HERE / "stanford_kinematics.cc"),
                        "-lm", "-o", str(tmp)], check=True)
        tmp.replace(lib)
    so = ctypes.CDLL(str(lib))
    P = np.ctypeslib.ndpointer
    f32, i32, ul = (P(dtype=np.float32, flags="C_CONTIGUOUS"), P(dtype=np.int32, flags="C_CONTIGUOUS"),
                    P(dtype=np.uint64, flags="C_CONTIGUOUS"))
    so.ch_reset.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_float, f32, ul]
    so.ch_step.argtypes = [ctypes.c_int, ctypes.c_void_p, f32, ctypes.c_float, ctypes.c_float, f32, ul, f32]
    so.ch_servo.argtypes = [ctypes.c_int, f32, f32, ctypes.c_float, ctypes.c_int, f32, i32]
    return so


def firmware_command(cmd_si) -> np.ndarray:
    """(n, 3) vx, vy m/s and wz rad/s -> the gait's command as ``set_velocity`` makes it (mm/s, rad/s, clamped)."""
    c = np.asarray(cmd_si, dtype=np.float32).reshape(-1, 3)
    out = np.empty_like(c)
    out[:, 0] = np.clip(c[:, 0] * np.float32(1000.0), -VXY_MAX_MM_S, VXY_MAX_MM_S)
    out[:, 1] = np.clip(c[:, 1] * np.float32(1000.0), -VXY_MAX_MM_S, VXY_MAX_MM_S)
    out[:, 2] = np.clip(c[:, 2], -WZ_MAX, WZ_MAX)
    return out


def moving(cmd_fw) -> np.ndarray:
    c = np.asarray(cmd_fw).reshape(-1, 3)
    return (np.abs(c[:, 0]) > MOVE_MM_S) | (np.abs(c[:, 1]) > MOVE_MM_S) | (np.abs(c[:, 2]) > MOVE_WZ)


def _mask(mask, n):
    if mask is None:
        return None
    m = np.ascontiguousarray(np.asarray(mask, dtype=np.int32).reshape(n))
    return m


class Gait:
    """``n`` robots' firmware gaits and servo writes (numpy, CPU)."""

    def __init__(self, n: int, offsets_deg, height_mm: float = HEIGHT_MM, clearance_mm: float = CLEARANCE_MM,
                 scale_dd: float = SCALE_DD, mul_first: bool = False):
        self.so = library()
        self.n = int(n)
        self.height, self.clearance = float(height_mm), float(clearance_mm)
        self.scale_dd, self.mul_first = float(scale_dd), int(bool(mul_first))
        self.off = np.ascontiguousarray(np.asarray(offsets_deg, dtype=np.float32).reshape(12))
        self.state = np.zeros((self.n, 12), dtype=np.float32)
        self.ticks = np.zeros(self.n, dtype=np.uint64)
        self.reset()

    def reset(self, mask=None) -> None:
        m = _mask(mask, self.n)
        self.so.ch_reset(self.n, None if m is None else m.ctypes.data, self.height, self.state, self.ticks)

    def step(self, cmd_fw, mask=None) -> np.ndarray:
        """One gait tick (``cmd_fw``: (n, 3) mm/s, mm/s, rad/s, as ``firmware_command`` gives): the feet (n, 4, 3) mm
        (rows of robots not stepped are zeros)."""
        m = _mask(mask, self.n)
        cmd = np.ascontiguousarray(np.broadcast_to(np.asarray(cmd_fw, dtype=np.float32), (self.n, 3)))
        feet = np.zeros((self.n, 12), dtype=np.float32)
        self.so.ch_step(self.n, None if m is None else m.ctypes.data, cmd, self.height, self.clearance, self.state,
                        self.ticks, feet)
        return feet.reshape(self.n, 4, 3)

    def servo(self, feet) -> tuple[np.ndarray, np.ndarray]:
        """(servo degrees before the offsets (k, 12), firmware ticks (k, 12)) for feet (k, 4, 3) mm."""
        f = np.ascontiguousarray(np.asarray(feet, dtype=np.float32).reshape(-1, 12))
        k = f.shape[0]
        sdeg = np.zeros((k, 12), dtype=np.float32)
        ticks = np.zeros((k, 12), dtype=np.int32)
        self.so.ch_servo(k, f, self.off, self.scale_dd, self.mul_first, sdeg, ticks)
        return sdeg, ticks


def neutral_feet(height_mm: float = HEIGHT_MM, n: int = 1) -> np.ndarray:
    f = np.zeros((n, 4, 3), dtype=np.float32)
    f[:, :, 2] = height_mm
    return f


def run_ticks(cmd_si, ticks: int, contract: Contract | None = None, **kw) -> np.ndarray:
    """One robot from the gait's reset: the firmware ticks (ticks, 12) under a constant command (m/s, rad/s)."""
    c = Contract.load() if contract is None else contract
    g = Gait(1, c.offsets_deg, **kw)
    cmd = firmware_command(cmd_si)
    out = []
    for _ in range(int(ticks)):
        out.append(g.servo(g.step(cmd))[1][0])
    return np.array(out)


# ------------------------------------------------------------------------------------------------- in simulation
CLASSIC_META = {"kind": "walk", "omni": True, "history": 0, "limits": True, "stand_offset_abd_deg": 0.0,
                "classic": True}
CLASSIC_SCALE = (0.25, 0.5, 0.5) * 4      # unused by the driver (the env's action scale for the observation's units)


class ClassicPolicy:
    """Stands in for an exported policy in the evaluators: ``meta`` and ``scale`` build the plant (the walker's
    evaluation plant without its stance offset: the classic stands at the calibrated stand), and calling it returns zero
    actions, which ``ClassicDriver`` ignores."""

    def __init__(self, device: str = "cuda:0"):
        self.meta = dict(CLASSIC_META)
        self.scale = CLASSIC_SCALE
        self.device = device

    def __call__(self, obs):
        import torch
        return torch.zeros(obs.shape[0], 12, device=obs.device)


class ClassicDriver:
    """Replaces an env's servo action term's decoder with the firmware gait (module docstring). The command is read
    from the env's ``twist`` term at each policy step (the evaluators write it before ``env.step``). Records per env:
    ``exc_bounds_deg`` (n, 12) the largest excess of a target over ``limits.decoder_bounds("walk")``, ``exc_band_deg``
    (n, 12) over the walking band (``BAND_DEG``, about the calibrated stand), ``max_step_deg`` (n, 12) the largest
    change between targets 20 ms apart (the decoder's step limit is ``STEP_LIMIT_DEG``), ``out_steps`` (n,) policy steps
    with any target outside the bounds, ``steps`` (n,); the same once the walk is ``STEADY_MS`` old (``steady_*``: the
    start from the stand excluded)."""

    def __init__(self, env, clamp: bool = False, height_mm: float = HEIGHT_MM, clearance_mm: float = CLEARANCE_MM):
        import torch
        from mp4_recovery.gpu.mdp import TorchBounds, servo_action
        from mp4_recovery.limits import decoder_bounds
        self.env = env
        self.term = t = servo_action(env)
        assert not t._shifted, "the classic runs at the calibrated stand (no stance offset)"
        self.dev = env.device
        self.n = env.num_envs
        self.contract = t.tc.c
        self.gait = Gait(self.n, self.contract.offsets_deg, height_mm, clearance_mm)
        self.clamp = bool(clamp)
        self.bounds_meta = decoder_bounds("walk", contract=self.contract)
        self.bounds = TorchBounds(self.bounds_meta, t.tc.stand_phys)
        self.band = torch.deg2rad(torch.tensor(BAND_DEG, dtype=torch.float32, device=self.dev))
        self.t_ms = np.zeros(self.n, dtype=np.int64)          # the walk's clock (ms; ticks at multiples of 15)
        self.walking = np.zeros(self.n, dtype=bool)
        self.tick_rel = np.zeros((self.n, 12), dtype=np.float32)   # the target in force (policy rel, tick centre)
        self.sched = None                                     # (n, 20, 12) model targets per substep
        self.prev_end = None
        self.reset_stats()
        self._orig_reset = t.reset
        t.process_actions = self.process_actions
        t.apply_actions = self.apply_actions
        t.reset = self.reset
        self.reset(None)
        env.classic_driver = self

    def reset_stats(self) -> None:
        import torch
        z = lambda *s: torch.zeros(*s, device=self.dev)   # noqa: E731
        self.acc = {k: {"exc_bounds": z(self.n, 12), "exc_band": z(self.n, 12), "max_step": z(self.n, 12),
                        "out_steps": z(self.n), "steps": z(self.n)} for k in ("all", "steady")}

    def reset(self, env_ids=None) -> None:
        self._orig_reset(env_ids)
        ids = np.arange(self.n) if env_ids is None else np.asarray(
            env_ids.cpu().numpy() if hasattr(env_ids, "cpu") else env_ids, dtype=np.int64).reshape(-1)
        mask = np.zeros(self.n, dtype=bool)
        mask[ids] = True
        self.gait.reset(mask)
        self.t_ms[mask] = 0
        self.walking[mask] = False
        self.tick_rel[mask] = 0.0
        if self.prev_end is not None:
            self.prev_end[torch_ids(ids, self.dev)] = 0.0

    def _rel_of_ticks(self, ticks) -> np.ndarray:
        return self.contract.rel_from_ticks(ticks).astype(np.float32)

    def process_actions(self, actions) -> None:
        import torch
        t = self.term
        t._raw[:] = actions
        cmd = firmware_command(self.env.command_manager.get_term("twist").vel_command_b.detach().cpu().numpy())
        dec = int(t._decimation)
        rel = np.empty((self.n, dec, 12), dtype=np.float32)
        for i in range(dec):
            T = self.t_ms + i
            start = ~self.walking & moving(cmd)
            if start.any():                       # the walk starts from the stand: stanford_gait_reset, then tick 0 now
                self.gait.reset(start)
                self.walking |= start
                self.t_ms[start] = -i            # the walk's own clock: its tick 0 at this substep
                T = self.t_ms + i
            fire = self.walking & (T % TICK_MS == 0)
            if fire.any():
                feet = self.gait.step(cmd, fire)
                _, ticks = self.gait.servo(feet[fire])
                self.tick_rel[fire] = self._rel_of_ticks(ticks)
            rel[:, i] = self.tick_rel
        self.t_ms += dec
        r = torch.as_tensor(rel, device=self.dev)                         # policy rel = physical rel (no offset)
        flat = r.view(-1, 12)
        if self.clamp:
            flat = torch.maximum(torch.minimum(flat, self.band), -self.band)
            flat = self.bounds.clamp_rel(flat)
        _, phys = t.tc.quantise_rel(flat)                                  # the dd of each tick, as the decoder sends
        phys = phys.view(self.n, dec, 12)
        self._record(phys)
        self.sched = t.tc.stand + phys
        end = phys[:, -1]
        t.prev_rel = end
        t.applied_action = end / t.scale
        t.target_model = self.sched[:, -1]
        self._sub = 0

    def _record(self, phys) -> None:
        import torch
        flat = phys.reshape(-1, 12)
        exb = torch.rad2deg((self.bounds.clamp_rel(flat) - flat).abs().view(self.n, -1, 12).amax(1))
        exn = torch.rad2deg((flat.abs() - self.band).clamp(min=0.0).view(self.n, -1, 12).amax(1))
        end = phys[:, -1]
        if self.prev_end is None:
            self.prev_end = torch.zeros_like(end)
        stp = torch.rad2deg((end - self.prev_end).abs())
        self.prev_end = end.clone()
        steady = torch.as_tensor(self.walking & (self.t_ms >= STEADY_MS), device=self.dev)
        for key, w in (("all", torch.ones_like(steady)), ("steady", steady)):
            a, wf = self.acc[key], w.float()
            a["exc_bounds"] = torch.maximum(a["exc_bounds"], exb * wf[:, None])
            a["exc_band"] = torch.maximum(a["exc_band"], exn * wf[:, None])
            a["max_step"] = torch.maximum(a["max_step"], stp * wf[:, None])
            a["out_steps"] += ((exb > 1e-6).any(-1) & w).float()
            a["steps"] += wf

    def apply_actions(self) -> None:
        t = self.term
        i = t._substep % t._decimation
        from mp4_recovery.gpu.mdp import SAMPLE_EVERY
        if i and i % SAMPLE_EVERY == 0:
            t._push(t.measured_rel())
        t._substep += 1
        tgt = self.sched[:, i] if self.sched is not None else t.target_model
        t.target_model = tgt
        t._entity.set_joint_position_target(tgt[:, t._jidx], joint_ids=t._tj)
        t._entity.set_tendon_len_target(tgt[:, t._kidx], tendon_ids=t._tt)

    def stats(self, ids) -> dict:
        """The walk caps' excursions over envs ``ids`` (deg): per servo kind the largest, and the share of policy steps
        with a target outside the walk bounds."""
        import torch
        ids = torch_ids(ids, self.dev)
        out = {}
        for key, pre in (("all", ""), ("steady", "steady_")):
            a = self.acc[key]
            for kind, cols in (("abd", [0, 3, 6, 9]), ("hip", [1, 4, 7, 10]), ("knee", [2, 5, 8, 11])):
                out[f"{pre}bounds_exc_{kind}_deg"] = float(a["exc_bounds"][ids][:, cols].max())
                out[f"{pre}band_exc_{kind}_deg"] = float(a["exc_band"][ids][:, cols].max())
                out[f"{pre}step20ms_max_{kind}_deg"] = float(a["max_step"][ids][:, cols].max())
            out[f"{pre}bounds_exc_per_servo_deg"] = [round(float(v), 2) for v in a["exc_bounds"][ids].max(0).values]
            out[f"{pre}share_steps_outside_bounds"] = float(a["out_steps"][ids].sum() / a["steps"][ids].sum().clamp(min=1))
        out["steady_after_ms"] = STEADY_MS
        out["step_limit_deg"] = STEP_LIMIT_DEG
        out["clamped"] = self.clamp
        return out


def torch_ids(ids, dev):
    import torch
    return torch.as_tensor(np.asarray(ids, dtype=np.int64), device=dev)


def describe(clamp: bool, height_mm: float = HEIGHT_MM, clearance_mm: float = CLEARANCE_MM) -> dict:
    """What an evaluation's record names about the classic controller."""
    return {"controller": "classic: firmware-dev Stanford trot (classic_gait.py)", "sources": SOURCES,
            "source_sha256": source_hash(), "height_mm": height_mm, "clearance_mm": clearance_mm,
            "tick_ms": TICK_MS, "command_clamp": {"vxy_mm_s": VXY_MAX_MM_S, "wz_rad_s": WZ_MAX},
            "clamped_to_walk_caps": bool(clamp)}
