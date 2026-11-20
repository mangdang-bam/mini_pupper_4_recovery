"""`watch` (chunked saves, markers, --disarm), `gains` and `scale` against the fake ESP32."""
import json
import os
import threading
import time

import numpy as np
import pytest
from fake_esp32 import FakeEsp32
from robot_helpers import C

from mp4_rl.getup.mp4link import cli
from mp4_rl.getup.mp4link import protocol as p

KFF = p.PARAM_ID["kff_current"]


@pytest.fixture
def fake(fixed_calibration):
    f = FakeEsp32()
    f.offsets = {i + 1: float(v) for i, v in enumerate(C.offsets_deg)}
    f.pos = [float(v) for v in C.stand_dd]
    for s in range(1, 13):
        f.params[s][KFF] = 0.001
    yield f
    f.close()


@pytest.fixture
def keys(monkeypatch):
    """A pipe standing in for the terminal: write lines to it."""
    r, w = os.pipe()
    monkeypatch.setattr(cli, "_stdin_fd", lambda: r)
    yield lambda text: os.write(w, text.encode())
    os.close(r)
    os.close(w)


def run_watch(fake, log, *extra, seconds=20):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rc", cli.main(
        ["--port", fake.port, "watch", "--seconds", str(seconds), "--log", str(log), *extra])))
    t.start()
    return t, out


def test_watch_saves_every_chunk_whole_and_records_markers(fake, keys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "WATCH_SAVE_S", 0.3)
    log = tmp_path / "w.npz"
    t, out = run_watch(fake, log)
    time.sleep(0.6)
    keys("m\n")
    time.sleep(0.5)
    mid = dict(np.load(log))                                  # a chunk written while the watch runs
    assert json.loads(str(mid["meta"]))["stop_reason"] == "running"
    assert mid["markers"].shape == (1, 2)
    keys("m\n")
    time.sleep(0.3)
    keys("\n")
    t.join(5)
    assert not t.is_alive() and out["rc"] == 0
    d = dict(np.load(log))
    meta = json.loads(str(d["meta"]))
    assert meta["stop_reason"] == "enter"
    assert len(d["pos_dd"]) > len(mid["pos_dd"])
    m = d["markers"]
    assert m.shape == (2, 2) and m[0, 0] < m[1, 0]
    k = m[:, 1].astype(int)
    assert 0 <= k[0] < k[1] < len(d["t_rx"])
    assert np.all(d["t_rx"][k] <= m[:, 0])                     # the marker's frame had arrived by then
    assert not (tmp_path / "w.npz.tmp").exists()
    assert np.all(d["sent_enable"] == 0) and not fake.armed_seen


def test_watch_q_ends_and_a_log_without_extension_is_npz(fake, keys, tmp_path):
    t, out = run_watch(fake, tmp_path / "w")
    time.sleep(0.5)
    keys("q\n")
    t.join(5)
    assert out["rc"] == 0
    d = dict(np.load(tmp_path / "w.npz"))
    assert json.loads(str(d["meta"]))["stop_reason"] == "q" and d["markers"].shape == (0, 2)


@pytest.mark.parametrize("extra, flags", [((), p.SA_DISARM), (("--no-disarm",), 0)])
def test_watch_disarm_flag_on_every_frame(fake, tmp_path, extra, flags):
    assert cli.main(["--port", fake.port, "watch", "--seconds", "0.3", "--log", str(tmp_path / "w.npz"), *extra]) == 0
    assert fake.frames and all(f[1] == flags and f[2] == 0 for f in fake.frames)
    assert json.loads(str(np.load(tmp_path / "w.npz")["meta"]))["disarm"] == (not extra)


def test_gains_reads_every_register_without_writing(fake, capsys, tmp_path):
    for s in range(1, 13):
        for name, v in cli.GAINS.items():
            fake.params[s][p.PARAM_ID[name]] = v
        fake.params[s][12] = 0.0123                            # pwm_scale
    fake.params[5][KFF] = 0.00022                              # the flash gain on one servo
    before = {s: dict(d) for s, d in fake.params.items()}
    assert cli.main(["--port", fake.port, "gains", "--out", str(tmp_path / "g.json")]) == 0
    out = capsys.readouterr().out
    assert fake.params == before and not fake.armed_seen
    rows = {line.split()[1]: line for line in out.splitlines() if line[:1].isdigit()}
    assert set(rows) == set(p.ALL_PARAMS)
    assert "kff_current !" in out and "kp_position !" not in out and "pwm_scale !" in out
    assert "0.00022" in rows["kff_current"] and "0.0123" in rows["pwm_scale"] and rows["pwm_scale"].split()[3] == "0.001"
    assert "pwm_scale answers on 12 servos" in out
    g = json.loads((tmp_path / "g.json").read_text())
    assert g["registers"]["pwm_scale"]["3"]["bits"] == "%08x" % np.float32(0.0123).view(np.uint32)
    assert cli.main(["--port", fake.port, "gains", "--hex"]) == 0
    assert "3a83126f" in capsys.readouterr().out            # 0.001 as float32


def test_gains_shows_a_register_the_firmware_refuses_as_missing(fake, capsys):
    for s in range(1, 13):
        del fake.params[s][12]                                 # an ESP32 or servo firmware without pwm_scale
    assert cli.main(["--port", fake.port, "gains"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "pwm_scale" in x and x[:1].isdigit())
    assert line.split()[-12:] == ["-"] * 12


def test_scale_reads_every_range_and_fails_on_a_mismatch(fake, capsys):
    for s in range(1, 13):
        fake.params[s][p.RANGE_PARAM] = 310.0
    assert cli.main(["--port", fake.port, "scale"]) == 0
    out = capsys.readouterr().out
    assert out.count(" ok") == 12 and "no RANGE flag" in out
    fake.params[8][p.RANGE_PARAM] = 300.0
    fake.extra_flags = p.ST_RANGE
    assert cli.main(["--port", fake.port, "scale"]) == 1
    cap = capsys.readouterr()
    assert "MISMATCH" in cap.out and "ARM refused" in cap.out and "servos [8]" in cap.err


def test_motion_refused_while_the_firmware_reports_range(fake, capsys):
    fake.extra_flags = p.ST_RANGE
    assert cli.main(["--port", fake.port, "centre", "--seconds", "0.1", "--clear"]) == 1
    assert "range_position_deg" in capsys.readouterr().err and not fake.armed_seen


def test_gains_marks_pwm_scale_only_when_it_differs_from_0_001(fake, capsys):
    for s in range(1, 13):
        for name, v in {**cli.GAINS, "pwm_scale": cli.PWM_SCALE}.items():
            fake.params[s][p.PARAM_ID[name]] = v
    assert cli.main(["--port", fake.port, "gains"]) == 0
    out = capsys.readouterr().out
    assert "!" not in out.split("- = no answer")[0]
    fake.params[9][p.PARAM_ID["pwm_scale"]] = 0.002
    assert cli.main(["--port", fake.port, "gains"]) == 0
    assert "pwm_scale !" in capsys.readouterr().out
