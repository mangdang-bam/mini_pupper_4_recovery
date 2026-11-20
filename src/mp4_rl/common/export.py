#!/usr/bin/env python
"""Export a training checkpoint as a robot policy directory: ``policy.npz`` (numpy = torch to 1e-5), the run's options
in ``config.json``, and byte copies of the calibration and the joint limits the policy was exported for
(``sku2_contract.json``, ``sku2_joint_limits.json``: the robot's files in ``mp4_sim``; the runner reads the calibration
from the folder).

    python -m mp4_rl.common.export skills/getup/checkpoints/sku2/getup/model_7543.pt runs/exports/getup

The joint limits and the stance offset come from the run's options (``limits``, ``stand_offset_abd_deg``) unless
``--limits`` / ``--no-limits`` / ``--stand-offset-abd`` say otherwise; with limits the meta gets the decoder's clamps
(``bounds.decoder_bounds(kind)``, with the run's ``widen_limits_deg``). ``--from-policy DIR``: no checkpoint: a copy of an
exported policy with the limits and offset written into its meta (a stand change without retraining):

    python -m mp4_rl.common.export --from-policy skills/walker/policies/sku2 runs/exports/walker --limits \
        --stand-offset-abd 12

The target filter's alpha (meta ``action_filter_alpha``) is the run's option ``filter_alpha`` (the source policy's meta with
``--from-policy``; 0 when absent) unless ``--filter-alpha`` says otherwise. The servo model (meta ``actuator``) is the
run's option ``actuator`` (the source's meta with ``--from-policy``; md300 when absent) unless ``--actuator`` says otherwise.
The run's legs and feet (options ``legs``, ``foot``, ``foot_radius_m``) and their geometry go into meta ``legs``
(``mp4_sim.robot.model.legs_meta``; with ``--from-policy`` the source's legs, rewritten for the installed model).
``--contract FILE``: another calibration (a robot calibrated anew, ``tools/make_contract.py``) for the bounds, the meta's
``contract_sha256`` and the folder's copy; default the robot's file in ``mp4_sim``.

``--robot``: the robot to export for (default: the run's option ``robot``, with ``--from-policy`` the source policy's
meta, else sku2). Its calibration, joint limits and hip envelope make the bounds; its calibration and limits are copied
into the folder (``sku1_contract.json``, ``sku1_joint_limits.json`` for the SKU1); the meta records it (``robot``, the
``legs`` block) when it is not sku2. A policy may be exported for another robot than it was trained on (e.g. a SKU2
policy for the SKU1, to evaluate or fine-tune it there): the export says so.

    python -m mp4_rl.common.export --from-policy skills/walker/policies/sku2 runs/exports/walker_sku1 --robot sku1
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mp4_rl.common.policy_file import export, with_meta


def copy_robot_files(out: Path, contract=None, robot=None) -> None:
    """Byte copies of the calibration (``contract``, else the robot's) and the robot's joint limits into ``out``."""
    from mp4_sim import robots
    r = robots.get(robot)
    src = Path(contract) if contract is not None else r.contract_json
    shutil.copyfile(src, out / r.contract_json.name)
    shutil.copyfile(r.limits_json, out / r.limits_json.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="a checkpoint (or with --from-policy: the source policy directory)")
    ap.add_argument("out")
    ap.add_argument("--from-policy", action="store_true",
                    help="CKPT is an exported policy directory: copy it with --limits / --stand-offset-abd in the meta")
    ap.add_argument("--limits", action="store_true", default=None,
                    help="the decoder's clamps in the meta (default: the run's option 'limits'; with --from-policy, the "
                         "source policy's meta)")
    ap.add_argument("--no-limits", dest="limits", action="store_false")
    ap.add_argument("--stand-offset-abd", type=float, metavar="DEG",
                    help="the abductions' stand offset, deg outward (default: the run's option, else 0; with --from-policy, "
                         "the source policy's meta)")
    ap.add_argument("--filter-alpha", type=float, metavar="ALPHA",
                    help="the decoder's target filter alpha (default: the run's option 'filter_alpha', else 0; with "
                         "--from-policy, the source policy's meta)")
    ap.add_argument("--kind", choices=("walk", "getup"),
                    help="the policy's kind (default: the run's option 'kind', else walk)")
    ap.add_argument("--band-deg", type=float, nargs="+", metavar="DEG",
                    help="the action band about the stand: 1, 3 (abduction, hip, knee) or 12 values (default: the "
                         "run's option 'band_deg', else the walking band 20/60/60)")
    ap.add_argument("--actuator", choices=("md01", "md300"),
                    help="the servo model the policy was trained on (default: the run's option 'actuator', else md300; with "
                         "--from-policy, the source policy's meta)")
    ap.add_argument("--contract", metavar="FILE",
                    help="the calibration to export for (default: the robot's in mp4_sim)")
    from mp4_sim import robots
    robots.add_args(ap, default_help="the run's option, with --from-policy the source's meta, else sku2")
    a = ap.parse_args()
    from mp4_sim.actuators.choice import check_name
    from mp4_sim.calibration import Contract

    from mp4_rl.common.bounds import decoder_bounds
    from mp4_rl.paths import sha256
    ck, out = Path(a.ckpt), Path(a.out)
    if a.from_policy:
        meta = json.loads(str(np.load(ck / "policy.npz")["meta"]))
        src_robot = (meta.get("legs") or {}).get("robot") or meta.get("robot")
    else:
        cfg = json.loads((ck.parent / "config.json").read_text())
        opts = cfg["task"]["options"]
        src_robot = opts.get("robot")
    robot = robots.get(a.robot or src_robot).name
    if robot != robots.get(src_robot).name:
        print(f"NOTE exporting for robot {robot}: trained on {robots.get(src_robot).name}", flush=True)
    cpath = Path(a.contract) if a.contract else robots.get(robot).contract_json
    contract = Contract.load(cpath)
    if a.from_policy:
        kind = a.kind or meta.get("kind") or "walk"
        lim = bool(meta.get("limits")) if a.limits is None else a.limits
        off = float(meta.get("stand_offset_abd_deg", 0.0) or 0.0) if a.stand_offset_abd is None else a.stand_offset_abd
        alpha = float(meta.get("action_filter_alpha", 0.0) or 0.0) if a.filter_alpha is None else a.filter_alpha
        widen = ((meta.get("bounds") or {}).get("info") or {}).get("widen_limits_deg")
        bounds = decoder_bounds(kind, contract=contract, widen_limits_deg=widen, robot=robot).to_meta() \
            if lim else None
        actuator = check_name(a.actuator or meta.get("actuator") or "md300")
        extra = {}
        if meta.get("legs") or robot != robots.DEFAULT:
            from mp4_sim.robot.model import legs_meta, legs_of
            extra["legs"] = legs_meta(legs_of({**(meta.get("legs") or {}), "robot": robot}))
        if robot != robots.DEFAULT or meta.get("robot"):
            extra["robot"] = robot
        path = with_meta(ck, out, kind=kind, limits=lim, bounds=bounds, contract_sha256=sha256(cpath),
                         stand_offset_abd_deg=float(off), action_filter_alpha=float(alpha), actuator=actuator, **extra)
        if (ck / "config.json").exists():
            shutil.copy(ck / "config.json", out / "config.json")
        copy_robot_files(out, a.contract, robot)
        print("EXPORTED", path)
        return
    kind = a.kind or opts.get("kind") or "walk"
    lim = bool(opts.get("limits", False)) if a.limits is None else a.limits
    off = float(opts.get("stand_offset_abd_deg", 0.0) or 0.0) if a.stand_offset_abd is None else a.stand_offset_abd
    alpha = float(opts.get("filter_alpha", 0.0) or 0.0) if a.filter_alpha is None else a.filter_alpha
    bounds = decoder_bounds(kind, contract=contract, widen_limits_deg=opts.get("widen_limits_deg"),
                            robot=robot).to_meta() if lim else None
    from mp4_sim.robot.model import legs_meta, legs_of
    legs = legs_meta(legs_of({**opts, "robot": robot}))
    path = export(ck, out, opts["action_scale"], history=int(opts.get("history", 0) or 0), band_deg=a.band_deg if a.band_deg is not None else opts.get("band_deg"),
                  kind=kind, getup_phase_s=float(opts.get("phase_s", 0.0) or 0.0),
                  bounds=bounds, stand_offset_abd_deg=off, action_filter_alpha=alpha,
                  actuator=check_name(a.actuator or opts.get("actuator") or "md300"), legs=legs,
                  contract_json=cpath, leg_leg_contacts=bool(opts.get("leg_leg_contacts", False)),
                  display_collision=bool(opts.get("display_collision", False)))
    # only what evaluation needs: a run's config also names local paths
    (out / "config.json").write_text(json.dumps({"task": cfg["task"], "noise": cfg.get("noise", "measured")}, indent=1))
    copy_robot_files(out, a.contract, robot)
    print("EXPORTED", path)


if __name__ == "__main__":
    main()
