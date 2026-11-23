#!/usr/bin/env python3
"""Write the calibration file for another Mini Pupper v4 unit, or for this one after a recalibration.

The calibration holds the robot's offsets and the stand they give in the servos' units. Calibrate the robot, read the
offsets on the Pi, and make the file on any machine:

    python3 -m mp4_rl.deploy.mp4link offsets --json offsets.json
    .venv/bin/python tools/make_contract.py offsets.json --note "unit 2, calibrated 2027-01-10" \\
        --out my_robot/sku2_contract.json

Only ``offsets_deg``, ``stand_dd``, ``stand_ticks`` and ``calibration`` change; the file replaced is listed under
``earlier_calibrations``, so policies exported for it still pass the calibration check of ``runner check``.

**What must then be re-exported.** A policy's decoder bounds hold the limits of the potentiometer tracks measured from the
stand, so they belong to one calibration, and ``runner run`` refuses a policy whose bounds were made for another stand. Write
each policy's bounds again for the new calibration (the network itself is unchanged); the folder gets a copy of it:

    .venv/bin/python -m mp4_rl.common.export --from-policy mp4_rl/skills/walker/policies/sku2 my_robot/walker --contract my_robot/sku2_contract.json
    .venv/bin/python -m mp4_rl.common.export --from-policy mp4_rl/skills/getup/policies/sku2 my_robot/getup --contract my_robot/sku2_contract.json

and run ``runner check`` on the results. The joint limits (``mp4_sim``'s ``sku2_joint_limits.json``) were measured on one
unit and are kept. Where a stand moved far along a joint's track, the bounds can cut motions the policy learned: compare the
new bounds with the old ones, and retrain or re-horn the joint if a range shrank a lot.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from mp4_rl.common.policy import ACTION_SIZE, CONTRACT_JSON, Contract
from mp4_rl.paths import sha256


def build(base_path, offsets, note: str) -> dict:
    """The new contract's JSON dict from the base file and the robot's offsets."""
    base = Contract.load(base_path)
    new = np.asarray(offsets, dtype=float)
    if new.shape != (ACTION_SIZE,):
        raise ValueError(f"12 offsets expected, got {new.shape}")
    ticks = dataclasses.replace(base, offsets_deg=new).ticks_from_servo_deg(np.zeros(ACTION_SIZE))
    d = json.loads(Path(base_path).read_text())
    d["earlier_calibrations"] = d.get("earlier_calibrations", []) + [
        {"sha256": sha256(base_path), "note": note, "offsets_deg": [float(v) for v in base.offsets_deg]}]
    d.update(offsets_deg=[float(v) for v in new], stand_dd=[int(v) for v in base.dd_from_ticks(ticks)],
             stand_ticks=[int(v) for v in ticks], calibration=note)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("offsets_json", help="the file `mp4link offsets --json` wrote")
    ap.add_argument("--note", required=True, help="which robot, and when it was calibrated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(CONTRACT_JSON), help="the calibration to start from (default: the SKU2's in mp4_sim)")
    a = ap.parse_args(argv)
    robot = json.loads(Path(a.offsets_json).read_text())
    if not all(robot["offsets_stored"]):
        raise SystemExit(f"refused: not every offset is stored on the robot: {robot['offsets_stored']}")
    d = build(a.base, robot["offsets_deg"], a.note)
    if robot["stand_dd"] != d["stand_dd"] or robot["stand_ticks"] != d["stand_ticks"]:
        raise SystemExit(f"refused: the robot's stand {robot['stand_dd']} / {robot['stand_ticks']} is not the one derived here "
                         f"{d['stand_dd']} / {d['stand_ticks']}")
    Path(a.out).write_text(json.dumps(d, indent=1) + "\n")
    Contract.load(a.out)
    print(f"wrote {a.out}\nsha256 {sha256(a.out)}\noffsets_deg {d['offsets_deg']}\nstand_dd {d['stand_dd']}")
    print("now re-export the policies' bounds: see this tool's docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
