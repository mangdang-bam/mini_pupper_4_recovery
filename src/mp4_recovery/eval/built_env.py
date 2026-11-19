"""What a result file records about the env a tool built: an option proved by the built model, not by the flag that
should have set it.

``record(env, opt, attachment)``: the plant's total mass read from the built model (``sum(body_mass)`` per world, the
attachment's per-robot draw included, and the compiled ``mj_model``'s, without it), the masses file (name, resolved path, sha256, its stated total), the legs, the servo model, the attachment, the
contract and the limits files (path, sha256) and the decoder's clamps (deg from the stand, servo order).

``masses_arg``: a ``--masses-json`` value (an existing path, made absolute, or a file name in src/mp4_recovery/data/);
without torch, so the CPU tests can parse a tool's flags.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from mp4_recovery import paths


def masses_path(name: str | None) -> Path:
    """The file ``robot.sku2.masses`` reads for ``masses_json`` = ``name`` (None: the package's ``sku2_masses.json``)."""
    return paths.MASSES_JSON if name is None else paths.DATA / name


def masses_arg(text: str) -> str:
    """``--masses-json``: an existing file (returned absolute, so the tool's working directory does not matter) or a
    file name in src/mp4_recovery/data/ (returned as given, as the options have always named it)."""
    p = Path(text).expanduser()
    if p.is_file():
        return str(p.resolve())
    if (paths.DATA / text).is_file():
        return text
    raise argparse.ArgumentTypeError(f"no masses file {text!r} (a path, or a name in {paths.DATA})")


def masses_record(name: str | None) -> dict:
    p = masses_path(name)
    total = json.loads(p.read_text()).get("total_kg")
    return {"masses_json": name, "path": str(p), "sha256": paths.sha256(p), "file_total_kg": total}


def contract_path() -> Path:
    """The contract ``Contract.load()`` reads by default."""
    from mp4_recovery.contract import Contract
    return Path(Contract.load.__func__.__defaults__[0])


def files_record() -> dict:
    from mp4_recovery import limits
    c = contract_path()
    out = {"contract": {"path": str(c), "sha256": paths.sha256(c)},
           "limits_file": {"path": str(limits.LIMITS_JSON), "sha256": paths.sha256(limits.LIMITS_JSON)}}
    if limits.HIP_ENVELOPE_JSON.is_file():
        out["hip_envelope_file"] = {"path": str(limits.HIP_ENVELOPE_JSON),
                                    "sha256": paths.sha256(limits.HIP_ENVELOPE_JSON)}
    return out


def _np(t):
    import numpy as np
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def total_mass(env) -> dict:
    """``sum(body_mass)`` of every world of the built env (kg), and of the compiled model."""
    bm = _np(env.sim.model.body_mass)
    tot = bm.reshape(-1, bm.shape[-1]).sum(axis=-1)
    return {"sum_body_mass_kg": {"min": round(float(tot.min()), 5), "max": round(float(tot.max()), 5),
                                 "mean": round(float(tot.mean()), 5)},
            "sum_body_mass_mj_model_kg": round(float(env.sim.mj_model.body_mass.sum()), 5)}


def _deg(t) -> list:
    import numpy as np
    return [round(float(v), 2) for v in np.degrees(np.asarray(_np(t), dtype=float)).ravel()]


def bounds_record(env) -> dict | None:
    """The clamps the env's decoder applies (deg from the stand, servo order), or None."""
    from mp4_recovery.gpu.mdp import servo_action
    b = servo_action(env).bounds
    if b is None:
        return None
    return {"lo_deg": _deg(b.lo), "hi_deg": _deg(b.hi), "calf_lo_deg": _deg(b.calf_lo), "calf_hi_deg": _deg(b.calf_hi),
            "hip_envelope_legs": sorted(int(k) for k in b.env)}


def _plain(v):
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def record(env, opt, attachment=None) -> dict:
    """The built env's record (see the module's docstring). ``attachment``: the one installed (default
    ``opt.attachment``, the get-up task's own option)."""
    att = attachment if attachment is not None else getattr(opt, "attachment", None)
    legs = getattr(opt, "legs", "sku_2")
    out = {**total_mass(env), "num_envs": int(env.num_envs),
           "masses": masses_record(getattr(opt, "masses_json", None)),
           "legs": {"legs": legs, "foot": getattr(opt, "foot", None), "foot_radius_m": getattr(opt, "foot_radius_m", None)},
           "actuator": {k: _plain(getattr(opt, k, None))
                        for k in ("actuator", "supply_v_range", "supply_drop_ohm_range", "r_scale_range")},
           "attachment": _plain(att),
           "limits": {"enabled": bool(getattr(opt, "limits", False)),
                      "widen_limits_deg": getattr(opt, "widen_limits_deg", None)},
           **files_record()}
    draw = getattr(env, "attachment_draw", None)
    if draw is not None:
        out["attachment_draw_mean"] = {k: round(float(_np(v).mean()), 5) for k, v in draw.items()}
    try:
        out["decoder_bounds"] = bounds_record(env)
    except (ImportError, KeyError, AttributeError, StopIteration, ValueError):
        out["decoder_bounds"] = "unavailable"
    return out


def line(rec: dict) -> str:
    """One printable line: total mass, legs, servo, attachment, contract and limits hashes."""
    m = rec["sum_body_mass_kg"]
    lg = rec["legs"]
    att = rec["attachment"]
    a = ("none" if att is None else
         f"h {att['height_mm']} mm r {att['radius_mm']} mm x {att['x_mm']} mm m {att['mass_g']} g")
    return (f"built env: total mass {m['min']:.5f}-{m['max']:.5f} kg (model without attachment "
            f"{rec['sum_body_mass_mj_model_kg']:.5f}; file {rec['masses']['masses_json'] or 'default'} "
            f"{rec['masses']['file_total_kg']}) legs {lg['legs']}"
            + f"/{lg['foot']} {lg['foot_radius_m']}"
            + f" actuator {rec['actuator']['actuator']} supply {rec['actuator']['supply_v_range']} attachment {a}"
            f" limits {rec['limits']['enabled']} contract {rec['contract']['sha256'][:8]}"
            f" limits_file {rec['limits_file']['sha256'][:8]}")
