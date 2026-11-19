"""compare_run's plant flags: parsed in every mode, passed to the simulation, recorded in the result (no simulation:
the env build is replaced by a stub)."""
import json
from pathlib import Path

import numpy as np
import pytest

import compare_run as tc  # noqa: E402
from mp4_recovery import paths

DATA = Path(__file__).resolve().parents[1] / "data" / "compare_run"
MASSES = "sku2_masses.json"           # a name in the package's data/


def _parse(argv, monkeypatch):
    seen = {}
    for mode in ("stream", "predict", "floor", "obs"):
        monkeypatch.setattr(tc, f"mode_{mode}", lambda a, m=mode: seen.setdefault("a", a) and 0)
    assert tc.main(argv) == 0 or "a" in seen
    return seen["a"]


PLANT = ["--legs", "sku_2", "--foot", "sphere", "--foot-radius", "0.006", "--masses-json", MASSES, "--actuator", "md300",
         "--supply-v", "10.5", "12.6", "--supply-drop-ohm", "0.1", "0.3", "--r-scale", "1.0", "1.4"]


@pytest.mark.parametrize("head", [["stream", "L.npz", "--stream", "S.npz"], ["predict", "--stream", "S.npz"],
                                  ["floor", "RUN"], ["obs", "RUN"]])
def test_every_mode_parses_the_plant_flags(head, monkeypatch):
    a = _parse(head + ["--policy", "P"] + PLANT, monkeypatch)
    p = tc.plant_of(a)
    assert p == {"actuator": {"supply_v_range": (10.5, 12.6), "supply_drop_ohm_range": (0.1, 0.3),
                              "r_scale_range": (1.0, 1.4), "actuator": "md300"},
                 "legs": {"legs": "sku_2", "foot": "sphere", "foot_radius_m": 0.006}, "masses_json": MASSES}


def test_unset_flags_leave_the_policys_plant(monkeypatch):
    a = _parse(["stream", "L.npz", "--stream", "S.npz", "--policy", "P"], monkeypatch)
    p = tc.plant_of(a)
    assert p["masses_json"] is None and p["legs"] == {"legs": None, "foot": None, "foot_radius_m": None}
    assert "actuator" not in p["actuator"] and all(v is None for v in p["actuator"].values())


def test_masses_json_path_made_absolute_and_unknown_refused(monkeypatch, tmp_path):
    f = tmp_path / "m.json"
    f.write_text((paths.DATA / MASSES).read_text())
    a = _parse(["predict", "--stream", "S.npz", "--policy", "P", "--masses-json", str(f)], monkeypatch)
    assert a.masses_json == str(f.resolve())
    with pytest.raises(SystemExit):
        tc.main(["predict", "--stream", "S.npz", "--policy", "P", "--masses-json", str(tmp_path / "none.json")])
    with pytest.raises(SystemExit):
        tc.main(["predict", "--stream", "S.npz", "--policy", "P", "--legs", "newer"])


def test_build_passes_the_plant_to_getup_eval(monkeypatch):
    from mp4_recovery.eval import built_env
    from mp4_recovery.eval import getup_eval as ge
    got = {}

    def build(pdir, robots, seed, **kw):
        got.update(kw, pdir=pdir)
        return "env", "opt"
    monkeypatch.setattr(ge, "build", build)
    monkeypatch.setattr(built_env, "record", lambda env, opt: {"legs": {"legs": "sku_2"}})
    monkeypatch.setattr(built_env, "line", lambda rec: "stub")
    plant = {"actuator": {"actuator": "md300", "supply_v_range": None}, "legs": {"legs": "sku_2", "foot": None,
                                                                                "foot_radius_m": None},
             "masses_json": MASSES}
    env, opt, rec = tc._build("P", 4, 1, "cpu", plant)
    assert rec == {"legs": {"legs": "sku_2"}}
    assert got["actuator"] == plant["actuator"] and got["legs"] == plant["legs"] and got["masses_json"] == MASSES
    tc._build("P", 4, 1, "cpu", {"actuator": tc.servo_flags(None)})          # no plant flags: nothing forced
    assert got["actuator"] is None and got["legs"] is None and got["masses_json"] is None


def test_stream_result_records_the_plant(monkeypatch, tmp_path):
    log = np.load(DATA / "fake_stream_back_x1.npz")
    meta = json.loads(str(log["meta"]))
    stand = np.asarray(meta["stand"], float)
    sent = np.asarray(log["sent_pos_dd"], float)
    stream = tmp_path / "s_x1p00.npz"
    np.savez(stream, delta_dd=sent - stand, period_s=0.02)
    seen = {}

    def sim_held(*args, plant=None, **kw):
        seen["plant"] = plant
        return {"duty_cap": 0.99, "actuator": "md300", "built_env": {"legs": {"legs": "sku_2"}, "masses": MASSES}}
    monkeypatch.setattr(tc, "sim_held", sim_held)
    monkeypatch.setattr(tc, "sim_report", lambda *a, **k: {})
    monkeypatch.setattr(tc, "compare_stream", lambda r, s, slow: {"groups": {"hip": {"pass": True, "ratio_flag": False}},
                                                                  "servos": {}})
    monkeypatch.setattr(tc, "print_stream", lambda c: None)
    out = tmp_path / "o.json"
    tc.main(["stream", str(DATA / "fake_stream_back_x1.npz"), "--stream", str(stream), "--policy", "P",
             "--legs", "sku_2", "--actuator", "md300", "--masses-json", MASSES, "--json", str(out)])
    r = json.loads(out.read_text())
    assert seen["plant"]["legs"]["legs"] == "sku_2" and seen["plant"]["masses_json"] == MASSES
    assert r["plant_flags"]["legs"]["legs"] == "sku_2" and r["plant_flags"]["actuator"]["actuator"] == "md300"
    assert r["plant_flags"]["masses_json"] == MASSES and r["built_env"]["legs"]["legs"] == "sku_2"
