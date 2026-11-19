"""Installing joint limits (tools/install_limits.py), the recorded hash, and the thigh stops read from the limits file."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from mp4_recovery import limits as L
from mp4_recovery import paths
from mp4_recovery.contract import Contract

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import install_limits as I  # noqa: E402

C = Contract.load()


def test_the_packaged_file_matches_its_recorded_hash_and_holds_no_thigh_stop():
    """No leg has a thigh-stop key, so no thigh gets a model range; the abduction and knee-fold ranges are the file's
    own."""
    assert paths.sha256(L.LIMITS_JSON) == L.recorded_sha256() == L.LIMITS_SHA256
    j = json.loads(L.LIMITS_JSON.read_text())
    assert all(L.THIGH_STOP_KEY not in d for d in j["legs"])
    assert L.thigh_stops_deg() == {}
    jr = L.joint_ranges()
    assert [k for k in jr if k.endswith("_thigh")] == []
    phys = L.physical_ranges()
    for d in j["legs"]:
        assert phys[d["leg"]]["knee_fold"] == tuple(float(v) for v in d["knee_fold_range"])


@pytest.mark.parametrize("name", ["walker", "getup"])
def test_every_policy_exports_the_bounds_it_holds(name):
    meta = json.loads(str(np.load(paths.POLICIES / name / "policy.npz")["meta"]))
    stored = meta["bounds"]
    widen = stored["info"].get("widen_limits_deg")
    fresh = L.decoder_bounds(meta.get("kind") or "walk", widen_limits_deg=widen).to_meta()
    for b in (stored, fresh):
        b["info"] = {k: v for k, v in b["info"].items() if k != "limits_sha256"}
    assert json.dumps(fresh, sort_keys=True) == json.dumps(stored, sort_keys=True)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A copy of the packaged limits, envelope and hash, which ``limits`` reads instead."""
    d = tmp_path / "data"
    d.mkdir()
    for f in (L.LIMITS_JSON, L.LIMITS_SHA256_FILE, L.HIP_ENVELOPE_JSON):
        shutil.copy(f, d / f.name)
    monkeypatch.setattr(L, "LIMITS_JSON", d / L.LIMITS_JSON.name)
    monkeypatch.setattr(L, "LIMITS_SHA256_FILE", d / L.LIMITS_SHA256_FILE.name)
    monkeypatch.setattr(L, "HIP_ENVELOPE_JSON", d / L.HIP_ENVELOPE_JSON.name)
    return d


def modified(tmp_path, change):
    j = json.loads(L.LIMITS_JSON.read_text())
    change(j)
    p = tmp_path / "new_limits.json"
    p.write_text(json.dumps(j, indent=1) + "\n")
    return p


def leg(j, name):
    return next(d for d in j["legs"] if d["leg"] == name)


def test_an_installed_file_loads_and_its_stops_apply(data, tmp_path, capsys):
    def change(j):
        leg(j, "FR")["thigh_stop_deg_backward"] = None
        leg(j, "FL")["thigh_stop_deg_backward"] = 30.0
        leg(j, "RR")["abd_out_range"] = [-10.0, 12.0]
    new = modified(tmp_path, change)
    env = tmp_path / "env.json"
    e = json.loads(L.HIP_ENVELOPE_JSON.read_text())
    e["envelope"]["FR"]["hi"] = [v - 0.01 for v in e["envelope"]["FR"]["hi"]]
    env.write_text(json.dumps(e))
    assert I.main(["--limits", str(new), "--hip-envelope", str(env), "--data", str(data)]) == 0
    assert (data / L.LIMITS_JSON.name).read_bytes() == new.read_bytes()
    assert (data / L.HIP_ENVELOPE_JSON.name).read_bytes() == env.read_bytes()
    assert L.recorded_sha256() == paths.sha256(new)
    assert L.load() == json.loads(new.read_text())
    assert L.thigh_stops_deg() == {"FL": 30.0}
    stand = dict(zip(C.joint_order, C.stand_joint_rad))["FL_thigh"]
    assert L.joint_ranges()["FL_thigh"] == (stand - float(np.radians(30.0)), stand + 2.0 * np.pi)   # FL: model + = forward
    assert L.legs()["RR"]["abd_out_range"] == [-10.0, 12.0]
    assert L.hip_envelope()["FR"]["hi"] == e["envelope"]["FR"]["hi"]
    assert "thigh stops (deg backward): {'FL': 30.0}" in capsys.readouterr().out


def test_a_file_changed_without_installing_is_refused(data, tmp_path):
    new = modified(tmp_path, lambda j: leg(j, "FR").update(knee_fold_range=[-80, 38]))
    shutil.copy(new, data / L.LIMITS_JSON.name)
    with pytest.raises(ValueError, match="is not the recorded"):
        L.load()
    assert I.main(["--limits", str(data / L.LIMITS_JSON.name), "--data", str(data)]) == 0   # installed in place
    assert L.load()["legs"][0]["knee_fold_range"] == [-80, 38]


@pytest.mark.parametrize("change, why", [
    (lambda j: j["legs"].pop(), "one entry for each"),
    (lambda j: leg(j, "FL").pop("signs_dd_to_phys"), "missing"),
    (lambda j: leg(j, "RL").update(thigh_stop_deg_backward="52"), "thigh_stop_deg_backward"),
    (lambda j: leg(j, "FR").update(knee_fold_range=[1.0, 2.0]), "lo < hi"),
])
def test_bad_files_are_refused_and_nothing_changes(data, tmp_path, capsys, change, why):
    before = {f.name: f.read_bytes() for f in data.iterdir()}
    assert I.main(["--limits", str(modified(tmp_path, change)), "--data", str(data)]) == 2
    assert why in capsys.readouterr().err
    assert {f.name: f.read_bytes() for f in data.iterdir()} == before
