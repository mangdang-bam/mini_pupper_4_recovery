"""The robot's measured joint limits in the model and the decoder.

Source: ``data/sku2_joint_limits.json``, its sha256 recorded in ``data/sku2_joint_limits.sha256`` (``load`` refuses a
file that differs: install a new one with ``tools/install_limits.py``, which copies it and records its hash). Per leg, in
physical terms (deg from the calibrated stand): the abduction's range by hand (``abd_out_range``, + = outward) and under
power (``abd_out_range_powered``), the knee relative to the thigh (``knee_fold_range``, by hand), the potentiometers'
track ends, the signs from the servos' dd direction to the physical directions, the hand ranges as the model's joint
ranges (``model_joint_range_rad``: ``*_hip`` = abduction, ``*_calf`` = knee - thigh), and optionally
``thigh_stop_deg_backward``: a measured stop on the thigh, deg backward from the stand, at every abduction (absent or
null: none).

- ``joint_ranges()``: the model's ``*_hip`` and ``*_calf`` ranges at the narrower of hand and powered on each side (the
  motor sometimes pressed further than the hand), and a ``*_thigh`` range for each leg with a thigh stop
  (``thigh_stops_deg``). A model joint is its stand (``stand_joint_rad``) + radians(the dd-direction value), the
  physical value times ``signs_dd_to_phys``.
- ``apply_limits(spec, margin_deg)``: those ranges on an ``MjSpec`` (the joints limited), each side narrowed by the
  margin.
- ``decoder_bounds(mode)``: the decoder's clamps (``contract.Bounds``) in the model's coordinates of the calibrated
  stand. Abduction: "walk" from ``WALK_ABD_MARGIN_DEG`` short of the inward stop to ``WALK_ABD_OUT_CAP_DEG`` outward (or
  the outward stop less the margin if nearer); "getup" the range less ``GETUP_MARGIN_DEG`` each side. Every hip and knee
  servo at least ``TRACK_MARGIN_DEG`` inside its potentiometer's track ends (``widen_limits_deg``: that far past them
  instead, for the first training step); the knee servo also so that the knee - thigh angle stays inside the ``*_calf``
  range less the mode's margin, given the hip target. ``hip_envelope`` (get-up only; ``sku2_hip_envelope.json``): the
  hip's chassis stop towards the body's middle as a function of the abduction.

Training directly inside the limits never learns to flip from the back, so the get-up's first step widens them by
``widen_limits_deg`` and the later steps train inside them. The exported meta records the widening
(``bounds.info.widen_limits_deg``), and ``eval_plant`` reads it back, so an evaluation of such a policy cannot drop it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mp4_recovery import paths
from mp4_recovery.contract import (ABDUCTION, ACTION_SIZE, LEG_SETS, TRACK_DD, Bounds, Contract, contract_tendon_coefs,
                                   leg_set_name)

LEGS = ("FR", "FL", "RR", "RL")
LIMITS_JSON = paths.DATA / "sku2_joint_limits.json"
LIMITS_SHA256_FILE = paths.DATA / "sku2_joint_limits.sha256"     # the installed limits file's sha256 (install_limits.py)
HIP_ENVELOPE_JSON = paths.DATA / "sku2_hip_envelope.json"
MODES = ("walk", "getup")
WALK_ABD_MARGIN_DEG = 3.0          # walking: the abduction this far short of its inward stop
WALK_ABD_OUT_CAP_DEG = 20.0        # walking: at most this far outward from the calibrated stand
GETUP_MARGIN_DEG = 2.0             # get-up: every measured range less this, each side
TRACK_MARGIN_DEG = 20.0            # every servo's target this far inside its potentiometer's track ends
THIGH_STOP_KEY = "thigh_stop_deg_backward"


def recorded_sha256() -> str:
    """The packaged limits file's recorded sha256 (``LIMITS_SHA256_FILE``)."""
    return LIMITS_SHA256_FILE.read_text().split()[0]


LIMITS_SHA256 = recorded_sha256()


def load(path=None) -> dict:
    """The limits file (the packaged copy, its hash checked against the recorded one, unless ``path``)."""
    p = LIMITS_JSON if path is None else Path(path)
    if path is None and paths.sha256(p) != recorded_sha256():
        raise ValueError(f"{p}: sha256 {paths.sha256(p)} is not the recorded {recorded_sha256()} "
                         f"({LIMITS_SHA256_FILE.name}; install limits with tools/install_limits.py)")
    return json.loads(p.read_text())


def contract_in_force() -> Path:
    """The contract file ``Contract.load()`` reads by default."""
    return Path(Contract.load.__func__.__defaults__[0])


def limits_contract_check(allow_mismatch: bool = False, j: dict | None = None) -> dict:
    """The packaged limits' calibration against the contract in force: the limits file's ``contract_sha256_prefix``
    (the contract it was traced at: the raw file's sha256, 12 hex) and the same prefix of
    ``contract_in_force()``. Raises ``ValueError`` on a mismatch unless ``allow_mismatch`` (limits traced at another
    calibration, used knowingly). Returns the record for a run's config."""
    import hashlib
    j = load() if j is None else j
    want = str(j.get("contract_sha256_prefix") or "")
    path = contract_in_force()
    have = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = {"limits_contract_sha256_prefix": want or None, "contract_in_force": str(path),
           "contract_in_force_sha256_prefix": have[:12], "match": bool(want) and have.startswith(want),
           "allow_limits_contract_mismatch": bool(allow_mismatch)}
    if not rec["match"] and not allow_mismatch:
        raise ValueError(f"the packaged limits were traced at contract {want or '(none recorded)'}, the contract in force "
                         f"({path}) is {have[:12]}: install limits fitted at this calibration, or pass "
                         f"--allow-limits-contract-mismatch")
    return rec


def legs(j: dict | None = None) -> dict:
    j = load() if j is None else j
    return {leg["leg"]: leg for leg in j["legs"]}


def thigh_stops_deg(j: dict | None = None) -> dict:
    """Per leg with one: the thigh's measured stop, deg backward from the stand (``thigh_stop_deg_backward``)."""
    return {leg: float(d[THIGH_STOP_KEY]) for leg, d in legs(j).items() if d.get(THIGH_STOP_KEY) is not None}


def physical_ranges(j: dict | None = None) -> dict:
    """Per leg (deg from the stand, physical directions): ``abd_out`` (inward stop < 0 < outward stop) at the narrower
    of hand and powered on each side, and ``knee_fold`` (hand)."""
    out = {}
    for leg, d in legs(j).items():
        hand, pw = d["abd_out_range"], d.get("abd_out_range_powered") or d["abd_out_range"]
        out[leg] = {"abd_out": (max(hand[0], pw[0]), min(hand[1], pw[1])),
                    "knee_fold": tuple(float(v) for v in d["knee_fold_range"])}
    return out


def _model(stand: float, sign: float, phys: tuple) -> tuple:
    a, b = stand + np.radians(sign * phys[0]), stand + np.radians(sign * phys[1])
    return (float(min(a, b)), float(max(a, b)))


def joint_ranges(j: dict | None = None, contract: Contract | None = None, hand_only: bool = False) -> dict:
    """The model's joint ranges (rad, absolute): ``{"FR_hip": (lo, hi), "FR_calf": (lo, hi), "FR_thigh": (lo, hi), ...}``. ``hand_only``:
    the hand ranges (what the file's ``model_joint_range_rad`` holds)."""
    j = load() if j is None else j
    c = Contract.load() if contract is None else contract
    stand = dict(zip(c.joint_order, c.stand_joint_rad))
    phys = physical_ranges(j)
    out = {}
    for leg, d in legs(j).items():
        s = d["signs_dd_to_phys"]
        abd = tuple(d["abd_out_range"]) if hand_only else phys[leg]["abd_out"]
        out[f"{leg}_hip"] = _model(stand[f"{leg}_hip"], s["abd_out"], abd)
        out[f"{leg}_calf"] = _model(stand[f"{leg}_calf"], s["knee_fold"], phys[leg]["knee_fold"])
    for leg, stop in thigh_stops_deg(j).items():     # the free side is left to the servo's track ends
        li = LEGS.index(leg)
        back = -float(c.signs[3 * li + 1]) * float(legs(j)[leg]["signs_dd_to_phys"]["hip_fwd"]) < 0   # model + = backward
        st = stand[f"{leg}_thigh"]
        r = float(np.radians(stop))
        out[f"{leg}_thigh"] = (st - 2.0 * np.pi, st + r) if back else (st - r, st + 2.0 * np.pi)
    return out


def apply_limits(spec, margin_deg: float = 0.0, ranges: dict | None = None) -> dict:
    """Limit the ``*_hip`` (abduction) and ``*_calf`` (knee - thigh) joints of an SKU2 ``MjSpec`` to ``ranges``
    (default ``joint_ranges()``), each side narrowed by ``margin_deg``. Returns the ranges set."""
    import mujoco
    ranges = joint_ranges() if ranges is None else ranges
    m = np.radians(float(margin_deg))
    out = {}
    for name, (lo, hi) in ranges.items():
        lo, hi = lo + m, hi - m
        if not lo < hi:
            raise ValueError(f"{name}: the margin {margin_deg} deg leaves no range")
        jt = spec.joint(name)
        jt.range = [lo, hi]
        jt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        out[name] = (lo, hi)
    return out


def abd_out_sign(j: dict | None = None) -> tuple:
    """Per leg the model / servo-rel sign of an outward abduction, from the file (= ``contract.ABD_OUT_SIGN``)."""
    return tuple(float(legs(j)[leg]["signs_dd_to_phys"]["abd_out"]) for leg in LEGS)


def track_rel(contract: Contract | None = None, margin_deg: float = TRACK_MARGIN_DEG) -> tuple:
    """Per servo (servo order) the rel (rad) ``margin_deg`` inside the potentiometer's track ends (dd ``TRACK_DD``):
    (lo, hi). A negative margin is past the ends."""
    c = Contract.load() if contract is None else contract
    m = 10.0 * float(margin_deg)
    a = c.rel_from_dd(np.full(ACTION_SIZE, TRACK_DD[0] + m))
    b = c.rel_from_dd(np.full(ACTION_SIZE, TRACK_DD[1] - m))
    return np.minimum(a, b), np.maximum(a, b)


def hip_envelope(contract: Contract | None = None, path=None) -> dict | None:
    """The get-up's hip envelope, or None without the file."""
    p = HIP_ENVELOPE_JSON if path is None else Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["envelope"]


def decoder_bounds(mode: str, contract: Contract | None = None, j: dict | None = None, envelope: bool = True,
                   widen_limits_deg: float | None = None) -> Bounds:
    """The decoder's clamps for ``mode`` ("walk" or "getup"; module docstring). ``envelope``: get-up mode includes
    the hip envelope when it exists. ``widen_limits_deg``: hip and knee targets may go this far past their track ends."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    j = load() if j is None else j
    c = Contract.load() if contract is None else contract
    stand = np.asarray(c.stand_rel_rad, dtype=float)
    track_margin = TRACK_MARGIN_DEG if widen_limits_deg is None else -float(widen_limits_deg)
    t_lo, t_hi = track_rel(c, track_margin)
    lo, hi = stand + t_lo, stand + t_hi
    phys = physical_ranges(j)
    margin = WALK_ABD_MARGIN_DEG if mode == "walk" else GETUP_MARGIN_DEG
    sign = abd_out_sign(j)
    abd_phys = {}
    for li, leg in enumerate(LEGS):
        a_in, a_out = phys[leg]["abd_out"]
        a = (a_in + margin, min(a_out - margin, WALK_ABD_OUT_CAP_DEG) if mode == "walk" else a_out - margin)
        abd_phys[leg] = a
        m_lo, m_hi = _model(stand[ABDUCTION[li]], sign[li], a)
        lo[ABDUCTION[li]] = max(lo[ABDUCTION[li]], m_lo)
        hi[ABDUCTION[li]] = min(hi[ABDUCTION[li]], m_hi)
    jr = joint_ranges(j, c)
    m = np.radians(margin)
    calf_lo = np.array([jr[f"{leg}_calf"][0] + m for leg in LEGS])
    calf_hi = np.array([jr[f"{leg}_calf"][1] - m for leg in LEGS])
    env = hip_envelope(c) if (mode == "getup" and envelope) else None
    b = Bounds(lo=lo, hi=hi, kc=contract_tendon_coefs(c), calf_lo=calf_lo, calf_hi=calf_hi, hip_envelope=env,
               mode=mode, info={"limits_sha256": paths.sha256(LIMITS_JSON), "abd_margin_deg": margin,
                                "walk_abd_out_cap_deg": WALK_ABD_OUT_CAP_DEG if mode == "walk" else None,
                                "calf_margin_deg": margin, "track_margin_deg": track_margin,
                                "widen_limits_deg": widen_limits_deg,
                                "abd_out_deg": {k: [float(x) for x in v] for k, v in abd_phys.items()},
                                "contract_stand_dd": [int(v) for v in c.stand_dd],
                                "hip_envelope": None if env is None else "sku2_hip_envelope.json"})
    b.check()
    return b


def describe(limits: bool, mode: str = "walk", stand_offset_abd_deg: float = 0.0) -> dict:
    """What a run's config and an evaluation's record name about the limits."""
    return {"limits": bool(limits), "mode": mode, "stand_offset_abd_deg": float(stand_offset_abd_deg),
            "limits_json": str(LIMITS_JSON.relative_to(paths.REPO)),
            "limits_sha256": paths.sha256(LIMITS_JSON) if limits else None,
            "joint_ranges": joint_ranges() if limits else None}


def eval_legs(meta: dict, legs: str | None = None, foot: str | None = None,
              foot_radius_m: float | None = None) -> dict:
    """An evaluation's legs (``TerrainOptions`` fields ``legs``, ``foot``, ``foot_radius_m``; ``robot.sku2.Legs``): the
    flags when given, else the policy's meta's ``legs`` block, else the defaults."""
    m = meta.get("legs") or {}
    name = leg_set_name(legs or m.get("legs"))
    same = bool(m) and leg_set_name(m.get("legs")) == name
    out = {"legs": name, "foot": foot or (m.get("foot") if same else None) or "sphere"}
    r = foot_radius_m if foot_radius_m is not None else (m.get("foot_radius_m") if same and not foot else None)
    if r is not None:
        out["foot_radius_m"] = float(r)
    return out


def legs_flags(args) -> dict:
    """``--legs``, ``--foot``, ``--foot-radius`` of an evaluation tool (``add_eval_args``), for ``eval_legs``."""
    return {"legs": getattr(args, "legs", None), "foot": getattr(args, "foot", None),
            "foot_radius_m": getattr(args, "foot_radius", None)}


def eval_plant(meta: dict, limits: bool | None = None, stand_offset_abd_deg: float | None = None,
               actuator: dict | None = None, legs: str | None = None, foot: str | None = None,
               foot_radius_m: float | None = None) -> dict:
    """An evaluation's plant options (``TerrainOptions`` fields) for a policy: the flags when given, else the policy's
    meta (``limits``, ``stand_offset_abd_deg``, the bounds' ``widen_limits_deg``, ``actuator``, the legs:
    ``eval_legs``). ``actuator``: the servo flags (``servo_model.options_from_args``; its ``actuator`` key absent: the
    meta's, else md300)."""
    from mp4_recovery.actuators import choice as servo_model
    lim = bool(meta.get("limits", False)) if limits is None else bool(limits)
    off = float(meta.get("stand_offset_abd_deg", 0.0) or 0.0) if stand_offset_abd_deg is None \
        else float(stand_offset_abd_deg)
    info = (meta.get("bounds") or {}).get("info") or {}
    servo = {"actuator": servo_model.check_name(meta.get("actuator") or servo_model.NAME), "supply_v_range": None,
             "supply_drop_ohm_range": None, "r_scale_range": None}
    servo.update(actuator or {})
    return {"limits": lim, "stand_offset_abd_deg": off, "widen_limits_deg": info.get("widen_limits_deg"), **servo,
            **eval_legs(meta, legs, foot, foot_radius_m)}


def eval_flags(args) -> dict:
    """``eval_plant``'s flag arguments from ``add_eval_args``'s namespace (servo model and legs included)."""
    from mp4_recovery.actuators import choice as servo_model
    return {"limits": getattr(args, "limits", None), "stand_offset_abd_deg": getattr(args, "stand_offset_abd", None),
            "actuator": servo_model.options_from_args(args), **legs_flags(args)}


def add_eval_args(ap) -> None:
    """``--limits`` / ``--no-limits``, ``--stand-offset-abd``, the servo model's flags (``--actuator``, ``--supply-v``,
    ``--supply-drop-ohm``, ``--r-scale``) and the legs (``--legs``, ``--foot``, ``--foot-radius``) for an evaluation
    tool (``eval_plant``, ``eval_flags``)."""
    ap.add_argument("--limits", action="store_true", default=None,
                    help="the measured joint limits in the plant and the decoder (default: the policy's meta)")
    ap.add_argument("--no-limits", dest="limits", action="store_false",
                    help="no joint limits even if the policy's meta has them")
    ap.add_argument("--stand-offset-abd", type=float, metavar="DEG", default=None,
                    help="the abductions' stand shifted outward by DEG (default: the policy's meta, else 0)")
    from mp4_recovery.actuators import choice as servo_model
    servo_model.add_args(ap, evaluation=True)
    add_legs_args(ap)


def add_legs_args(ap, default_help: str = "the policy's meta, else sku_2") -> None:
    """``--legs``, ``--foot``, ``--foot-radius`` (``robot.sku2.Legs``)."""
    ap.add_argument("--legs", choices=LEG_SETS, default=None,
                    help=f"the robot's legs: sku_2 (59.75 mm shin) (default: {default_help})")
    ap.add_argument("--foot", choices=("sphere", "mesh"), default=None,
                    help="the feet: a sphere (--foot-radius) or the model's mesh (default: sphere)")
    ap.add_argument("--foot-radius", type=float, metavar="M", default=None,
                    help="the foot sphere's radius (m; default 0.0055)")
