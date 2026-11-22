"""How far each leg's hip goes towards the body's middle, per abduction bin, against what the robot has shown it can do.

Towards the middle is backward on the front legs and forward on the rear legs (``hip_chassis_side`` in
``mp4_sim``'s ``sku2_joint_limits.json``), in deg from the calibrated stand. The robot's logs give the proven reach, the
range each hip has already moved through on the robot (``getup/data/sku2_proven_reach.json``); ``ReachMeter`` sets a simulated policy against it.
A value is a bin's 99.5th percentile, from at least 20 samples. A bin without data has no proven reach, so anything
there counts as beyond it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mp4_rl.common import bounds as limits
from mp4_rl.common.policy import Contract

REACH_JSON = Path(__file__).resolve().parents[1] / "data" / "sku2_proven_reach.json"
ABD_BINS_DEG = (-25, -15, -10, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30)
PERCENTILE = 99.5
MIN_SAMPLES = 20


def bin_label(lo: float, hi: float) -> str:
    return f"[{lo:g},{hi:g})"


def from_dd(rel_dd_deg, leg_data: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(abduction outward, hip towards the middle), deg, each (..., 4), from the 12 servos' deg from the stand in the
    direction their dd grow (servo order)."""
    d = limits.legs() if leg_data is None else leg_data
    x = np.asarray(rel_dd_deg, dtype=float).reshape(*np.shape(rel_dd_deg)[:-1], 4, 3)
    abd = np.array([d[leg]["signs_dd_to_phys"]["abd_out"] for leg in limits.LEGS], dtype=float)
    mid = np.array([d[leg]["signs_dd_to_phys"]["hip_fwd"] * (-1.0 if d[leg]["hip_chassis_side"] == "backward" else 1.0)
                    for leg in limits.LEGS], dtype=float)
    return x[..., 0] * abd, x[..., 1] * mid


def from_rel(rel_rad, contract: Contract | None = None, leg_data: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``from_dd`` for physical rel (rad, relative to the calibrated stand, the policy's coordinates without an offset)."""
    c = Contract.load() if contract is None else contract
    return from_dd(-c.signs * np.degrees(np.asarray(rel_rad, dtype=float)), leg_data)


def per_bin(abd, mid) -> list:
    """The ``PERCENTILE`` of ``mid`` in each of ``ABD_BINS_DEG``'s bins, by ``abd``; None under ``MIN_SAMPLES``."""
    abd, mid = np.asarray(abd), np.asarray(mid)
    out = []
    for lo, hi in zip(ABD_BINS_DEG[:-1], ABD_BINS_DEG[1:]):
        v = mid[(abd >= lo) & (abd < hi)]
        out.append(round(float(np.percentile(v, PERCENTILE)), 1) if len(v) >= MIN_SAMPLES else None)
    return out


def top(values: list) -> float | None:
    v = [x for x in values if x is not None]
    return max(v) if v else None


def load_proven(path=None) -> dict:
    return json.loads((REACH_JSON if path is None else Path(path)).read_text())


class ReachMeter:
    """Collects commanded and measured poses (``add``) and reports them per leg against the proven reach (``report``)."""

    def __init__(self, proven: dict | None = None):
        self.proven = load_proven() if proven is None else proven
        self.c = Contract.load()
        self.legs = limits.legs()
        self.samples = {"commanded": [], "measured": []}

    def add(self, commanded_rel, measured_rel) -> None:
        """Physical rel (rad), (robots, 12), one call per step."""
        for kind, rel in (("commanded", commanded_rel), ("measured", measured_rel)):
            self.samples[kind].append(np.stack(from_rel(rel, self.c, self.legs), -1).astype(np.float32))

    def add_servo(self, servo, mask) -> None:
        """One step of a simulated ``ServoAction``, robots ``mask``: the target in force and the newest position."""
        if bool(mask.any()):
            commanded = servo.target_model[mask] - servo.tc.stand_phys
            measured = servo.pos_rel[mask] + (servo.off if servo.off.dim() == 1 else servo.off[mask])
            self.add(commanded.cpu().numpy(), measured.cpu().numpy())

    def report(self) -> dict:
        edges = list(zip(ABD_BINS_DEG[:-1], ABD_BINS_DEG[1:]))
        legs, beyond = {}, []
        for li, leg in enumerate(limits.LEGS):
            proven = self.proven["legs"][leg]["reach_deg"]
            row = {"proven_deg": proven, "proven_max_deg": top(proven)}
            for kind, batches in self.samples.items():
                a = np.concatenate(batches)[:, li] if batches else np.zeros((0, 2))
                vals = per_bin(a[:, 0], a[:, 1])
                row[f"{kind}_deg"] = vals
                row[f"{kind}_max_deg"] = top(vals)
                row[f"{kind}_peak_deg"] = round(float(a[:, 1].max()), 1) if len(a) else None
                for (lo, hi), v, p in zip(edges, vals, proven):
                    if v is not None and (p is None or v > p):
                        beyond.append({"leg": leg, "kind": kind, "abd_bin_deg": [lo, hi], "deg": v, "proven_deg": p,
                                       "excess_deg": None if p is None else round(v - p, 1)})
            legs[leg] = row
        beyond.sort(key=lambda b: (b["excess_deg"] is not None, -(b["excess_deg"] or 0.0)))
        return {"percentile": PERCENTILE, "min_samples": MIN_SAMPLES, "abd_bins_deg": list(ABD_BINS_DEG), "legs": legs,
                "beyond": beyond}


def print_report(rep: dict, out=print) -> None:
    for leg, r in rep["legs"].items():
        out(f"reach {leg}: commanded {r['commanded_max_deg']} (peak {r['commanded_peak_deg']}), measured "
            f"{r['measured_max_deg']} (peak {r['measured_peak_deg']}), proven {r['proven_max_deg']} deg towards the middle")
    out(f"reach beyond the proven: {len(rep['beyond'])}")
    for b in rep["beyond"]:
        out(f"  {b['leg']} {b['kind']} abduction {bin_label(*b['abd_bin_deg'])}: {b['deg']} deg, proven {b['proven_deg']}")


def add_args(ap) -> None:
    ap.add_argument("--reach", nargs="?", const=str(REACH_JSON), metavar="JSON",
                    help="report each leg's commanded and measured hip towards the body's middle against this proven "
                         "reach (default without a value: the packaged one)")
