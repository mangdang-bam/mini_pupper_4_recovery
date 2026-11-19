"""The plant options of the trainer, the evaluators and getup_stream: parsed, threaded into the options the env is built
from, and recorded (``eval.built_env``). The built env's own total mass is checked by the GPU smokes, not here."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mp4_recovery import paths
from mp4_recovery.eval import built_env

ROOT = Path(__file__).resolve().parents[2]
ATT = "35,19,0.043,43.2"


def masses_file(tmp_path, total=0.6305) -> Path:
    d = json.loads((paths.DATA / "sku2_masses.json").read_text())
    d["total_kg"] = total
    f = tmp_path / "masses_test.json"
    f.write_text(json.dumps(d))
    return f


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_masses_arg(tmp_path, monkeypatch):
    f = masses_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert built_env.masses_arg("masses_test.json") == str(f.resolve())     # a relative path, made absolute
    assert built_env.masses_arg("sku2_masses.json") == "sku2_masses.json"   # a name in data/
    with pytest.raises(Exception):
        built_env.masses_arg("no_such_masses.json")
    # robot.sku2.masses reads paths.DATA / name: an absolute path survives the join
    assert built_env.masses_path(str(f.resolve())) == f.resolve()
    assert built_env.masses_path(None) == paths.MASSES_JSON


def test_masses_record(tmp_path):
    f = masses_file(tmp_path, 0.6305)
    r = built_env.masses_record(str(f))
    assert r["file_total_kg"] == 0.6305 and r["sha256"] == paths.sha256(f) and r["path"] == str(f)


def test_record_fake_env(tmp_path):
    from mp4_recovery.gpu.attachment import Attachment   # torch only
    f = masses_file(tmp_path)
    bm = np.array([[0.0, 0.5, 0.1305], [0.0, 0.55, 0.1305]])
    env = SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(body_mass=bm),
                                              mj_model=SimpleNamespace(body_mass=bm[0])), num_envs=2)
    att = Attachment.parse(ATT)
    opt = SimpleNamespace(masses_json=str(f), legs="sku_2", foot="sphere", foot_radius_m=0.0055, actuator="md300",
                          supply_v_range=(10.5, 12.6), supply_drop_ohm_range=None, r_scale_range=None, limits=True,
                          widen_limits_deg=None)
    r = built_env.record(env, opt, att)
    assert r["sum_body_mass_kg"] == {"min": 0.6305, "max": 0.6805, "mean": 0.6555}
    assert r["sum_body_mass_mj_model_kg"] == 0.6305
    assert r["legs"] == {"legs": "sku_2", "foot": "sphere", "foot_radius_m": 0.0055}
    assert r["actuator"]["actuator"] == "md300" and r["actuator"]["supply_v_range"] == [10.5, 12.6]
    assert r["attachment"]["mass_g"] == (43.2, 43.2) and r["attachment"]["radius_mm"] == 19.0
    from mp4_recovery.contract import CONTRACT_JSON
    from mp4_recovery import limits
    assert r["contract"]["sha256"] == paths.sha256(CONTRACT_JSON)
    assert r["limits_file"]["sha256"] == paths.sha256(limits.LIMITS_JSON)
    assert r["limits"]["enabled"] is True
    assert r["decoder_bounds"] == "unavailable"            # a fake env has no decoder
    json.dumps(r)
    assert "0.6305" in built_env.line(r)


def test_train_masses_json(tmp_path):
    pytest.importorskip("mjlab")
    from mp4_recovery.gpu import train
    from mp4_recovery.gpu.getup_task import GetupOptions
    f = masses_file(tmp_path)
    assert train.task_options(train.parse_args([])).masses_json == GetupOptions.masses_json   # default unchanged
    o = train.task_options(train.parse_args(["--masses-json", str(f), "--legs", "sku_2", "--actuator", "md300",
                                             "--getup-attachment"]))
    assert o.masses_json == str(f) and o.legs == "sku_2" and o.actuator == "md300" and o.attachment is not None
    with pytest.raises(SystemExit):
        train.parse_args(["--masses-json", str(tmp_path / "missing.json")])


def test_terrain_eval_flags(tmp_path):
    pytest.importorskip("mjlab")
    from mp4_recovery.eval import terrain_eval as te
    f = masses_file(tmp_path)
    a = te.make_parser().parse_args(["--policy", "p", "--masses-json", str(f), "--attachment", ATT, "--legs", "sku_2",
                                     "--actuator", "md300"])
    assert a.masses_json == str(f) and a.attachment.height_mm == (35.0, 35.0) and a.attachment.x_mm == (43.0, 43.0)
    a = te.make_parser().parse_args(["--policy", "p"])
    assert a.masses_json is None and a.attachment is None


@pytest.mark.parametrize("mod", ["omni_eval", "push_eval"])
def test_attachment_flag(mod):
    pytest.importorskip("mjlab")
    m = importlib.import_module(f"mp4_recovery.eval.{mod}")
    assert m.make_parser().parse_args(["--policy", "p", "--attachment", ATT]).attachment.mass_g == (43.2, 43.2)
    assert m.make_parser().parse_args(["--policy", "p"]).attachment is None


def test_omni_build_passes_attachment(monkeypatch):
    """omni_eval's build hands the attachment to terrain_eval.build (which installs it last)."""
    pytest.importorskip("mjlab")
    from mp4_recovery.eval import omni_eval as oe
    seen = {}

    def fake_build(opt, render=False, seed=None, camera=None, attachment=None):
        seen["attachment"], seen["masses_json"] = attachment, opt.masses_json
        return SimpleNamespace()
    monkeypatch.setattr(oe.te, "build", fake_build)
    pol = SimpleNamespace(scale=tuple([0.1] * 12), meta={})
    args = oe.make_parser().parse_args(["--policy", "p", "--attachment", ATT, "--masses-json", "sku2_masses.json"])
    oe.build(args, pol, 4, 10.0)
    assert seen["attachment"].height_mm == (35.0, 35.0) and seen["masses_json"] == "sku2_masses.json"


def test_getup_stream_plant_flags(tmp_path, monkeypatch):
    gs = load_tool("getup_stream")
    f = masses_file(tmp_path)
    got = {}
    monkeypatch.setattr(gs, "main_policy", lambda a: got.setdefault("a", a) and 0)
    gs.main(["--policy", "p"])
    a = got.pop("a")
    assert gs.plant_kwargs(a) == {"actuator": {"supply_v_range": None, "supply_drop_ohm_range": None,
                                               "r_scale_range": None},
                                  "legs": {"legs": None, "foot": None, "foot_radius_m": None}, "masses_json": None}
    gs.main(["--policy", "p", "--legs", "sku_2", "--foot-radius", "0.0055", "--actuator", "md300", "--supply-v", "10.5",
             "12.6", "--masses-json", str(f)])
    a = got.pop("a")
    k = gs.plant_kwargs(a)
    assert k["actuator"] == {"actuator": "md300", "supply_v_range": (10.5, 12.6), "supply_drop_ohm_range": None,
                             "r_scale_range": None}
    assert k["legs"] == {"legs": "sku_2", "foot": None, "foot_radius_m": 0.0055} and k["masses_json"] == str(f)
    assert gs.plant_flags_record(a)["masses_json"] == str(f)
    with pytest.raises(SystemExit):
        gs.main(["--legs", "sku_2"])                              # --policy is required
