"""What a result file records about the env a tool built: an option proved by the built model, not by the flag that
should have set it.

``record(env, opt, attachment)``: the plant's total mass read from the built model (``sum(body_mass)`` per world, the
attachment's per-robot draw included, and the compiled ``mj_model``'s, without it), the masses file (name, its
``mp4_sim:`` name, sha256, its stated total), the legs, the servo model, the attachment, the robot model, calibration,
limits and hip envelope files (``mp4_sim:`` name, sha256), the decoder's clamps (deg from the stand, servo order) and
the packages' versions (``versions``).

``masses_arg``: a ``--masses-json`` value (an existing path, made absolute, or a file name in a robot's measured
data in ``mp4_sim``); without torch, so the CPU tests can parse a tool's flags.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from mp4_sim import paths as sim_paths
from mp4_sim import robots


def masses_path(name: str | None, robot=None) -> Path:
    """The file ``mp4_sim.robot.model.masses`` reads for ``masses_json`` = ``name`` (None: the robot's own)."""
    from mp4_sim.robot.model import masses_path as model_masses_path
    return model_masses_path(name, robot)


def masses_arg(text: str) -> str:
    """``--masses-json``: an existing file (returned absolute, so the tool's working directory does not matter) or a
    file name in the robot's measured data (returned as given, as the options have always named it)."""
    p = Path(text).expanduser()
    if p.is_file():
        return str(p.resolve())
    for r in robots.ROBOTS:
        if (robots.get(r).measured / text).is_file():
            return text
    where = ", ".join(sim_paths.name(robots.get(r).measured) for r in robots.ROBOTS)
    raise argparse.ArgumentTypeError(f"no masses file {text!r} (a path, or a name in {where})")


def masses_record(name: str | None, robot=None) -> dict:
    p = masses_path(name, robot)
    total = json.loads(p.read_text()).get("total_kg")
    return {"masses_json": name, "path": sim_paths.name(p), "sha256": sim_paths.sha256(p), "file_total_kg": total}


def contract_path(robot=None) -> Path:
    """The calibration ``Contract.load()`` reads by default for ``robot``."""
    return robots.get(robot).contract_json


def _file(p: Path) -> dict:
    return {"path": sim_paths.name(p), "sha256": sim_paths.sha256(p)}


def files_record(robot=None, legs=None) -> dict:
    """The robot's model, calibration, limits and hip envelope files (``mp4_sim:`` names and sha256)."""
    r = robots.get(robot)
    out = {"contract": _file(r.contract_json), "limits_file": _file(r.limits_json)}
    if r.hip_envelope_json.is_file():
        out["hip_envelope_file"] = _file(r.hip_envelope_json)
    out["model_file"] = _file(r.xml if legs is None else legs.xml)
    return out


def run_versions() -> dict:
    """The packages a run used: this package's and ``mp4_sim``'s version, and each one's git commit when it runs from
    a checkout (None otherwise)."""
    import subprocess
    from pathlib import Path

    import mp4_sim

    import mp4_rl

    def commit(mod):
        d = Path(mod.__file__).resolve().parent
        try:
            r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
            s = subprocess.run(["git", "-C", str(d), "status", "--porcelain", "--untracked-files=no", "."],
                               capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None, None
        if r.returncode != 0:
            return None, None
        return r.stdout.strip(), bool(s.stdout.strip())

    out = {}
    for name, mod in (("mp4_rl", mp4_rl), ("mp4_sim", mp4_sim)):
        c, dirty = commit(mod)
        out[name] = {"version": getattr(mod, "__version__", None), "commit": c, "dirty": dirty}
    return out


def versions() -> dict:
    """The packages' versions and git commits (``run_versions``)."""
    return run_versions()


def files_and_versions(opt=None) -> dict:
    """``files_record`` and ``versions`` for an evaluator that builds its env without ``record`` (a result file's
    provenance): the robot model, calibration, limits, masses and the packages."""
    from mp4_sim.robot.model import legs_of
    legs = legs_of(opt)
    out = files_record(legs.robot, legs)
    out["masses"] = masses_record(getattr(opt, "masses_json", None), legs.robot)
    out["versions"] = versions()
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
    from mp4_rl.common.mdp import servo_action
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
    from mp4_sim.robot.model import legs_of
    lg = legs_of(opt)
    legs = getattr(opt, "legs", "sku_2")
    out = {**total_mass(env), "num_envs": int(env.num_envs),
           "masses": masses_record(getattr(opt, "masses_json", None), lg.robot),
           "legs": {"legs": legs, "foot": getattr(opt, "foot", None), "foot_radius_m": getattr(opt, "foot_radius_m", None)},
           "actuator": {k: _plain(getattr(opt, k, None))
                        for k in ("actuator", "supply_v_range", "supply_drop_ohm_range", "r_scale_range")},
           "attachment": _plain(att),
           **({"robot": lg.robot} if lg.robot != robots.DEFAULT else {}),
           **({k: True for k in ("leg_leg_contacts", "display_collision") if getattr(opt, k, False)}),
           "limits": {"enabled": bool(getattr(opt, "limits", False)),
                      "widen_limits_deg": getattr(opt, "widen_limits_deg", None)},
           **files_record(lg.robot, lg), "versions": versions()}
    draw = getattr(env, "attachment_draw", None)
    if draw is not None:
        out["attachment_draw_mean"] = {k: round(float(_np(v).mean()), 5) for k, v in draw.items()}
    dz = getattr(env, "com_height_draw", None)
    if dz is not None:          # mp4_sim.envs.com_height: the shared CoM shift of the base and the hip modules (m)
        out["com_height_draw_m"] = {"min": round(float(_np(dz).min()), 5), "max": round(float(_np(dz).max()), 5)}
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
    rob = f"robot {rec['robot']} " if rec.get("robot") else ""
    return (f"built env: {rob}total mass {m['min']:.5f}-{m['max']:.5f} kg (model without attachment "
            f"{rec['sum_body_mass_mj_model_kg']:.5f}; file {rec['masses']['masses_json'] or 'default'} "
            f"{rec['masses']['file_total_kg']}) legs {lg['legs']}"
            + f"/{lg['foot']} {lg['foot_radius_m']}"
            + f" actuator {rec['actuator']['actuator']} supply {rec['actuator']['supply_v_range']} attachment {a}"
            f" limits {rec['limits']['enabled']} contract {rec['contract']['sha256'][:8]}"
            f" limits_file {rec['limits_file']['sha256'][:8]} model {rec['model_file']['sha256'][:8]}"
            f" mp4_sim {rec['versions']['mp4_sim']['version']}")
