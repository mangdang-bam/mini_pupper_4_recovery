"""The decoder's clamps from the robot's measured joint limits, and the evaluation tools' plant flags.

The limits themselves (the file, its recorded hash, the joint and physical ranges, the hip envelope, the model's joint
ranges) are the robot's measured data in ``mp4_sim.limits``; this module turns them into a policy's decoder bounds.

- ``decoder_bounds(mode)``: the decoder's clamps (``common.policy.Bounds``) in the model's coordinates of the calibrated
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

import numpy as np
from mp4_sim import limits as _sim_limits
from mp4_sim import paths as sim_paths
from mp4_sim import robots
from mp4_sim.limits import (  # noqa: F401  (re-exported for the tools)
    HIP_ENVELOPE_JSON,
    LEGS,
    LIMITS_JSON,
    LIMITS_SHA256,
    LIMITS_SHA256_FILE,
    THIGH_STOP_KEY,
    TRACK_MARGIN_DEG,
    abd_out_sign,
    apply_limits,
    hip_envelope,
    hip_envelope_path,
    joint_ranges,
    limits_contract_check,
    load,
    model_range,
    physical_ranges,
    recorded_sha256,
    thigh_stops_deg,
    track_rel,
)

from mp4_rl.common.policy import (
    ABDUCTION,
    LEG_SETS,
    Bounds,
    Contract,
    contract_tendon_coefs,
    leg_set_name,
)

legs = _sim_limits.legs          # the limits file's per-leg entries (re-exported)
MODES = ("walk", "getup")
WALK_ABD_MARGIN_DEG = 3.0          # walking: the abduction this far short of its inward stop
WALK_ABD_OUT_CAP_DEG = 20.0        # walking: at most this far outward from the calibrated stand
GETUP_MARGIN_DEG = 2.0             # get-up: every measured range less this, each side


def decoder_bounds(mode: str, contract: Contract | None = None, j: dict | None = None, envelope: bool = True,
                   widen_limits_deg: float | None = None, robot=None) -> Bounds:
    """The decoder's clamps for ``mode`` ("walk" or "getup"; module docstring) on ``robot`` (None: SKU2).
    ``envelope``: get-up mode includes the hip envelope when it exists. ``widen_limits_deg``: hip and knee targets may
    go this far past their track ends."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    r = robots.get(robot)
    j = load(robot=r) if j is None else j
    c = Contract.load(robot=r) if contract is None else contract
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
        m_lo, m_hi = model_range(stand[ABDUCTION[li]], sign[li], a)
        lo[ABDUCTION[li]] = max(lo[ABDUCTION[li]], m_lo)
        hi[ABDUCTION[li]] = min(hi[ABDUCTION[li]], m_hi)
    jr = joint_ranges(j, c)
    m = np.radians(margin)
    calf_lo = np.array([jr[f"{leg}_calf"][0] + m for leg in LEGS])
    calf_hi = np.array([jr[f"{leg}_calf"][1] - m for leg in LEGS])
    env = hip_envelope(c, robot=r) if (mode == "getup" and envelope) else None
    b = Bounds(lo=lo, hi=hi, kc=contract_tendon_coefs(c), calf_lo=calf_lo, calf_hi=calf_hi, hip_envelope=env,
               mode=mode, info={"limits_sha256": sim_paths.sha256(r.limits_json), "abd_margin_deg": margin,
                                "walk_abd_out_cap_deg": WALK_ABD_OUT_CAP_DEG if mode == "walk" else None,
                                "calf_margin_deg": margin, "track_margin_deg": track_margin,
                                "widen_limits_deg": widen_limits_deg,
                                "abd_out_deg": {k: [float(x) for x in v] for k, v in abd_phys.items()},
                                "contract_stand_dd": [int(v) for v in c.stand_dd],
                                "hip_envelope": None if env is None else hip_envelope_path(r).name})
    b.check()
    return b


def describe(limits: bool, mode: str = "walk", stand_offset_abd_deg: float = 0.0, robot=None) -> dict:
    """What a run's config and an evaluation's record name about the limits."""
    r = robots.get(robot)
    return {"limits": bool(limits), "mode": mode, "stand_offset_abd_deg": float(stand_offset_abd_deg),
            "limits_json": sim_paths.name(r.limits_json),
            "limits_sha256": sim_paths.sha256(r.limits_json) if limits else None,
            "joint_ranges": joint_ranges(robot=r) if limits else None}


def eval_legs(meta: dict, legs: str | None = None, foot: str | None = None,
              foot_radius_m: float | None = None, robot: str | None = None,
              leg_leg_contacts: bool | None = None, display_collision: bool | None = None) -> dict:
    """An evaluation's robot, legs and contacts (``TerrainOptions`` fields ``robot``, ``legs``, ``foot``,
    ``foot_radius_m``, ``leg_leg_contacts``, ``display_collision``; ``mp4_sim.robot.model.Legs``): the flags when
    given, else the policy's meta (its ``legs`` block, its ``robot``, ``leg_leg_contacts``, ``display_collision``),
    else the defaults. ``robot`` is in the result only when it is not SKU2, and the two contact options only when on."""
    m = meta.get("legs") or {}
    rob = robots.get(robot or m.get("robot") or meta.get("robot")).name
    name = leg_set_name(legs or m.get("legs"))
    same = bool(m) and leg_set_name(m.get("legs")) == name
    out = {"legs": name, "foot": foot or (m.get("foot") if same else None) or "sphere"}
    r = foot_radius_m if foot_radius_m is not None else (m.get("foot_radius_m") if same and not foot else None)
    if r is not None:
        out["foot_radius_m"] = float(r)
    if rob != robots.DEFAULT:
        out["robot"] = rob
    for k, flag in (("leg_leg_contacts", leg_leg_contacts), ("display_collision", display_collision)):
        if (bool(meta.get(k, False)) if flag is None else bool(flag)):
            out[k] = True
    return out


def legs_flags(args) -> dict:
    """``--robot``, ``--legs``, ``--foot``, ``--foot-radius``, ``--display-collision`` of an evaluation tool
    (``add_eval_args``), for ``eval_legs``."""
    out = {"legs": getattr(args, "legs", None), "foot": getattr(args, "foot", None),
           "foot_radius_m": getattr(args, "foot_radius", None)}
    for k in ("robot", "display_collision"):        # only when given
        if getattr(args, k, None) is not None:
            out[k] = getattr(args, k)
    return out


def eval_plant(meta: dict, limits: bool | None = None, stand_offset_abd_deg: float | None = None,
               actuator: dict | None = None, legs: str | None = None, foot: str | None = None,
               foot_radius_m: float | None = None, robot: str | None = None,
               display_collision: bool | None = None) -> dict:
    """An evaluation's plant options (``TerrainOptions`` fields) for a policy: the flags when given, else the policy's
    meta (``limits``, ``stand_offset_abd_deg``, the bounds' ``widen_limits_deg``, ``actuator``, the legs:
    ``eval_legs``). ``actuator``: the servo flags (``servo_model.options_from_args``; its ``actuator`` key absent: the
    meta's, else md300)."""
    from mp4_sim.actuators import choice as servo_model
    lim = bool(meta.get("limits", False)) if limits is None else bool(limits)
    off = float(meta.get("stand_offset_abd_deg", 0.0) or 0.0) if stand_offset_abd_deg is None \
        else float(stand_offset_abd_deg)
    info = (meta.get("bounds") or {}).get("info") or {}
    servo = {"actuator": servo_model.check_name(meta.get("actuator") or servo_model.NAME), "supply_v_range": None,
             "supply_drop_ohm_range": None, "r_scale_range": None}
    servo.update(actuator or {})
    return {"limits": lim, "stand_offset_abd_deg": off, "widen_limits_deg": info.get("widen_limits_deg"), **servo,
            **eval_legs(meta, legs, foot, foot_radius_m, robot, display_collision=display_collision)}


def eval_flags(args) -> dict:
    """``eval_plant``'s flag arguments from ``add_eval_args``'s namespace (servo model and legs included)."""
    from mp4_sim.actuators import choice as servo_model
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
    from mp4_sim.actuators import choice as servo_model
    servo_model.add_args(ap, evaluation=True)
    add_legs_args(ap)


def add_legs_args(ap, default_help: str = "the policy's meta, else sku_2") -> None:
    """``--robot``, ``--legs``, ``--foot``, ``--foot-radius`` (``mp4_sim.robot.model.Legs``) and ``--display-collision``
    / ``--no-display-collision``."""
    robots.add_args(ap, default_help=default_help.replace("sku_2", "sku2"))
    ap.add_argument("--legs", choices=LEG_SETS, default=None,
                    help=f"the robot's legs: sku_2 (59.75 mm shin) (default: {default_help})")
    ap.add_argument("--foot", choices=("sphere", "mesh"), default=None,
                    help="the feet: a sphere (--foot-radius) or the model's mesh (default: sphere)")
    ap.add_argument("--foot-radius", type=float, metavar="M", default=None,
                    help="the foot sphere's radius (m; default 0.0055)")
    ap.add_argument("--display-collision", dest="display_collision", action="store_true", default=None,
                    help="the SKU1's display as a collision box on the base (default: the policy's meta, else off)")
    ap.add_argument("--no-display-collision", dest="display_collision", action="store_false",
                    help="without the display box even if the policy's meta has it")


def refuse_provisional(kind: str, robot=None, j: dict | None = None) -> None:
    """A get-up is never trained or exported with bounds from a limits file marked ``provisional`` (a stand-in for
    limits not yet traced on the robot): ``ValueError``. The walker may use one."""
    j = load(robot=robot) if j is None else j
    if kind == "getup" and j.get("provisional"):
        raise ValueError(f"robot {robots.get(robot).name}: its joint limits are provisional ({j['provisional']}); "
                         "a get-up needs limits traced on the robot")
