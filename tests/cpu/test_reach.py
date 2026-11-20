"""The proven reach: its directions, its bins, the shipped file, and a simulated policy set against it."""
import numpy as np

from mp4_rl.common import bounds as limits
from mp4_rl.common.policy import Contract
from mp4_rl.getup.eval import reach

C = Contract.load()


def test_towards_the_middle_is_the_same_from_dd_and_from_rel():
    rng = np.random.default_rng(0)
    dd = C.stand_dd + rng.integers(-500, 500, (50, 12))
    abd_dd, mid_dd = reach.from_dd((dd - C.stand_dd) / 10.0)
    abd_rel, mid_rel = reach.from_rel(C.rel_from_dd(dd))
    assert np.abs(abd_dd - abd_rel).max() < 0.5 and np.abs(mid_dd - mid_rel).max() < 0.5


def test_directions_are_the_limits_file_and_the_chassis_side():
    leg_data = limits.legs()
    for li, leg in enumerate(limits.LEGS):
        s = leg_data[leg]["signs_dd_to_phys"]
        d = np.zeros(12)
        d[3 * li] = 10.0 * s["abd_out"]             # 10 deg outward
        d[3 * li + 1] = 20.0 * s["hip_fwd"]         # 20 deg forward
        abd, mid = reach.from_dd(d)
        assert abd[li] == 10.0 and abs(mid[li]) == 20.0
        assert mid[li] == (-20.0 if leg_data[leg]["hip_chassis_side"] == "backward" else 20.0)
    # the front-right thigh's stop, 52 deg backward: the model's joint limit is on the same side
    rel = np.zeros(12)
    rel[1] = np.radians(52.0)
    assert reach.from_rel(rel)[1][0] == 52.0


def test_per_bin_needs_its_samples():
    abd = np.r_[np.full(19, 0.5), np.full(200, 1.5), np.full(30, 40.0)]
    mid = np.r_[np.full(19, 9.0), np.arange(200.0), np.full(30, 99.0)]
    v = reach.per_bin(abd, mid)
    assert v[reach.ABD_BINS_DEG.index(0)] is None
    assert v[reach.ABD_BINS_DEG.index(1)] == round(float(np.percentile(np.arange(200.0), 99.5)), 1)
    assert sum(x is not None for x in v) == 1


def test_shipped_reach():
    d = reach.load_proven()
    assert d["abd_bins_deg"] == list(reach.ABD_BINS_DEG) and d["percentile"] == reach.PERCENTILE
    assert set(d["legs"]) == set(limits.LEGS)
    for r in d["legs"].values():
        assert len(r["reach_deg"]) == len(reach.ABD_BINS_DEG) - 1 and r["max_deg"] == reach.top(r["reach_deg"])
    assert d["legs"]["FR"]["max_deg"] < 55.0 < d["legs"]["FL"]["max_deg"]
    assert d["legs"]["RR"]["max_deg"] < 45.0 and d["legs"]["RL"]["max_deg"] < 25.0
    assert d["stalls"]["FR_thigh"]["runs"] == 18 and d["stalls"]["FR_calf"]["runs"] == 6 and len(d["stalls"]) == 2


def proven(value=None):
    bins = len(reach.ABD_BINS_DEG) - 1
    return {"legs": {leg: {"reach_deg": [value] * bins} for leg in limits.LEGS}}


def meter_with_rr(hip_forward_deg, abd_out_deg, p):
    """A meter fed one step of 100 robots, the rear-right leg held there, the rest at the stand."""
    m = reach.ReachMeter(p)
    s = limits.legs()["RR"]["signs_dd_to_phys"]
    dd = np.zeros((100, 12))
    dd[:, 6] = abd_out_deg * s["abd_out"]
    dd[:, 7] = np.linspace(0.0, hip_forward_deg, 100) * s["hip_fwd"]
    rel = -C.signs * np.radians(dd)                    # dd direction to rel: from_rel's inverse
    m.add(rel, 0.5 * rel)
    return m


def test_meter_lists_what_is_beyond_the_proven():
    rep = meter_with_rr(60.0, -12.0, proven(30.0)).report()
    rr = rep["legs"]["RR"]
    assert 59.0 < rr["commanded_max_deg"] <= 60.0 and rr["commanded_peak_deg"] == 60.0
    assert 29.0 < rr["measured_max_deg"] <= 30.0 and rr["proven_max_deg"] == 30.0
    rows = [b for b in rep["beyond"] if b["leg"] == "RR"]
    assert [b["kind"] for b in rows] == ["commanded"] and rows[0]["abd_bin_deg"] == [-15, -10]
    assert rows[0]["excess_deg"] == round(rows[0]["deg"] - 30.0, 1)


def test_meter_counts_a_bin_without_data_as_beyond():
    rep = meter_with_rr(5.0, -12.0, proven(None)).report()
    rows = [b for b in rep["beyond"] if b["leg"] == "RR" and b["kind"] == "measured"]
    assert rows[0]["proven_deg"] is None and rows[0]["excess_deg"] is None


def test_print_report(capsys):
    reach.print_report(meter_with_rr(60.0, -12.0, proven(30.0)).report())
    out = capsys.readouterr().out
    assert "reach RR: commanded" in out and "RR commanded abduction [-15,-10):" in out
