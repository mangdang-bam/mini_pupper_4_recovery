"""The walker trainer (``walker/train.py``) on the CPU: its command line, the range curriculum, the omni shares, the
attachment option, the fine-tuning recipes' argv, and AnchorPPO's anchor masks (``gpu/anchor_ppo.py``)."""
import math
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("rsl_rl")
import torch  # noqa: E402

from mp4_rl.common import anchor_ppo  # noqa: E402
from mp4_rl.skills.walker import task as wtask  # noqa: E402
from mp4_rl.skills.walker import train as tw  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = {"action_scale": [0.25, 0.5, 0.5] * 4, "history": 10}


def opts(*argv, step=0):
    a = tw.build_parser().parse_args(["--resume", "/x/model_1.pt", *argv])
    return a, tw.walker_options(a, SRC, step)


# ------------------------------------------------------------------------------------------------ the command line
def test_defaults_are_the_walkers_recipe():
    a, o = opts()
    assert (o.terrain, o.narrow_risers, o.stair_exit, o.max_init_level) == ("narrow", (0.025, 0.045), True, 9)
    assert (o.foot_softness, o.friction_range, o.tilt_limit) == ((0.005, 0.02), (0.4, 1.3), 1.0)
    assert (o.clearance, o.clearance_weight, o.stumble, o.progress, o.overspeed) == ("apex", -2.0, -0.5, 2.0, -2.0)
    assert (o.omni_promote_rel, o.heading, o.heading_law, o.heading_wz, o.vx_max) == (0.8, "loop", "pi", 0.4, 0.12)
    assert o.omni_mix == "default" and wtask.mix_shares(o) == (0.40, 0.40, 0.10, 0.10, 0.0)
    assert (o.actuator, o.legs, o.masses_json, o.limits, o.stand_offset_abd_deg) == ("md300", "sku_2",
                                                                                     "sku2_masses.json", False, 0.0)
    assert o.history == 10 and tuple(o.action_scale) == tuple(SRC["action_scale"])
    assert (a.anchor_mask, a.beta0, a.beta_hold, a.beta_decay, a.critic_warmup, a.lr0) == ("inrange", 5.0, 100000, 0,
                                                                                           30, 1e-4)
    # no curriculum unless asked: every range starts and ends at the parent's
    assert o.fwd_vx_range == (0.12, 0.12) and o.spin_wz_max == (1.0, 1.0) and o.omni_speed_max == (0.12, 0.12)
    assert o.push_range is None and o.attachment is None and not o.push_obs


def test_plant_flags():
    _, o = opts("--legs", "sku_2", "--foot", "mesh", "--actuator", "md01", "--supply-v", "10.5", "12.6",
                "--masses-json", "/abs/masses.json", "--limits", "--stand-offset-abd", "12",
                "--attachment-lidar", "0.5")
    assert (o.legs, o.foot, o.actuator, o.supply_v_range) == ("sku_2", "mesh", "md01", (10.5, 12.6))
    assert (o.masses_json, o.limits, o.stand_offset_abd_deg) == ("/abs/masses.json", True, 12.0)
    att = o.attachment
    assert (att.height_mm, att.mass_g, att.x_mm, att.radius_mm, att.present) == ((35.0, 35.0), (43.2, 43.2),
                                                                                 (43.0, 43.0), 19.0, 0.5)


def test_attachment_flags_exclusive_and_checked():
    with pytest.raises(SystemExit):
        opts("--attachment", "35,19,0.043,43.2", "--attachment-lidar", "0.5")
    with pytest.raises(SystemExit):
        opts("--attachment-lidar", "1.5")
    _, o = opts("--attachment", "35,19,0.043,43.2")
    assert o.attachment.present is None and o.attachment.height_mm == (35.0, 35.0)


def test_kick_flags():
    with pytest.raises(SystemExit):
        opts("--push-mps", "0", "0.6")                        # no interval
    with pytest.raises(SystemExit):
        opts("--anchor-push-mask", "1", "--push-mps", "0", "1", "--push-interval", "2", "5")   # no anchor
    _, o = opts("--anchor", "/x/model_1.pt", "--push-mps", "0", "0.6", "--push-interval", "2", "5", "--topple-mps",
                "0.6", "1.5", "--topple-interval", "8", "15", "--anchor-push-mask", "1")
    assert wtask.push_tiers(o) == ((0.0, 0.6, 2.0, 5.0), (0.6, 1.5, 8.0, 15.0))
    assert o.push_fall_grace == 2.0 and o.push_obs


def test_command_curriculum_flags():
    _, o = opts("--cmd-full", "--cmd-ramp", "300", "--omni-mix", "spin_up", step=6297 * 24)
    assert o.fwd_vx_range == (0.12, 0.20) and o.omni_speed_max == (0.12, 0.20)
    assert o.spin_wz_max == (1.0, 2.0) and o.turn_wz_max == (0.6, 1.0) and o.omni_wz == (0.6, 1.0)
    assert (o.cmd_ramp_iters, o.cmd_ramp_start_step) == (300, 6297 * 24)
    assert wtask.mix_shares(o) == wtask.OMNI_MIXES["spin_up"]
    _, o = opts("--spin-wz-end", "1.5", "--omni-shares", "1", "1", "1", "1", "1")
    assert o.spin_wz_max == (1.0, 1.5) and o.fwd_vx_range == (0.12, 0.12)
    assert wtask.mix_shares(o) == (1.0,) * 5
    with pytest.raises(SystemExit):
        opts("--spin-wz-end", "0.5")                          # below the parent's: not a widening


def test_ramp_fraction_and_ranges():
    s0 = 6297 * 24
    assert tw.STEPS_PER_ITER == 24
    assert wtask.ramp_fraction(s0, s0, 300) == 0.0
    assert wtask.ramp_fraction(s0 + 150 * 24, s0, 300) == pytest.approx(0.5)
    assert wtask.ramp_fraction(s0 + 1000 * 24, s0, 300) == 1.0
    assert wtask.ramp_fraction(0, s0, 300) == 0.0            # the first reset, before the counter is restored
    assert wtask.ramp_fraction(0, s0, 0) == 1.0              # no ramp: the end ranges at once
    assert wtask.lerp((1.0, 2.0), 0.25) == pytest.approx(1.25)


def recipe_argv(tmp_path, name, *args):
    """A recipe's command line, captured by a stand-in interpreter: (the parsed arguments, the walker's options)."""
    dump = tmp_path / "argv.txt"
    stub = tmp_path / "py"
    stub.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > {dump}\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    subprocess.run(["sh", str(REPO / "mp4_rl/skills/walker/recipes/sku2" / name), *args], check=True, cwd=REPO,
                   env={**os.environ, "PYTHON": str(stub)})
    argv = dump.read_text().splitlines()
    a = tw.build_parser().parse_args(argv[argv.index("mp4_rl.skills.walker.train") + 1:])
    return a, tw.walker_options(a, SRC, 6297 * 24)


def test_recipe_step1_argv(tmp_path):
    """The first step's command line parses and gives this robot's plant."""
    a, o = recipe_argv(tmp_path, "step1.sh", "7", "/abs/walker/model_6297.pt", "500")
    assert a.resume == a.anchor == "/abs/walker/model_6297.pt" and a.anchor_mask == "inrange"
    assert (a.iters, a.seed, a.envs, a.tag) == (500, 7, 1024, "step1")
    assert o.limits and o.stand_offset_abd_deg == 12.0 and o.attachment.present == 0.5
    assert o.spin_wz_max == (1.0, 2.0) and o.cmd_ramp_iters == 300 and o.omni_mix == "spin_up"
    assert wtask.push_tiers(o) == ((0.0, 0.6, 4.0, 8.0), (0.6, 1.5, 8.0, 15.0)) and o.push_obs
    assert a.anchor_push_mask == 2.0 and a.save_at == [300]
    assert wtask.push_start_steps(o) == (0, 6297 * 24 + 300 * 24)
    assert not a.allow_limits_contract_mismatch
    assert (o.legs, o.foot, o.actuator, o.supply_v_range, o.r_scale_range) == ("sku_2", "sphere", "md300", (10.5, 12.6),
                                                                              (0.95, 1.15))
    assert o.masses_json == "sku2_masses.json"


# ------------------------------------------------------------------------------------------------ the anchor masks
CMDS = torch.tensor([
    [0.10, 0.0, 0.0],      # 0 forward, in the parent's range
    [0.12, 0.0, 0.3],      # 1 forward at the top with a turn
    [0.20, 0.0, 0.0],      # 2 forward beyond
    [0.0, 0.10, 0.0],      # 3 sideways in range
    [0.0, -0.20, 0.0],     # 4 sideways beyond
    [-0.0848, 0.0848, 0.0],  # 5 a backward diagonal at 0.12
    [0.0, 0.0, 1.0],       # 6 spin at the parent's top
    [0.0, 0.0, -1.5],      # 7 spin beyond
    [0.0, 0.0, 0.0],       # 8 standing
    [0.10, 0.0, 0.8],      # 9 walking with a turn faster than the parent's walks
    [0.0, 0.0, 0.15],      # 10 a slow spin
])


def test_inrange_weights():
    w = anchor_ppo.inrange_weights(CMDS)
    assert w.tolist() == [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    w = anchor_ppo.inrange_weights(CMDS, range_vx=0.20, range_omni=0.20, range_wz=2.0, range_walk_wz=1.0)
    assert w.tolist() == [1] * 11


def test_nonforward_weights_unchanged():
    # the nonforward mask: backward, sideways / diagonal, spins in place above 0.1 rad/s
    assert anchor_ppo.nonforward_weights(CMDS).tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1]


def _alg(mask, **kw):
    return SimpleNamespace(anchor_mask=mask, anchor_push_mask=kw.pop("push", None),
                           anchor_range={**anchor_ppo.RANGE_DEFAULT, **kw})


def test_anchor_terms_masked_mean():
    """The masked term is sum_i w_i mse_i / N: beta keeps its per-sample meaning."""
    n = CMDS.shape[0]
    mean = torch.zeros(n, 12)
    mu = torch.arange(n, dtype=torch.float32)[:, None].expand(n, 12).clone()
    alg = _alg("inrange")
    alg.anchor_weights = lambda c, p=None: anchor_ppo.AnchorPPO.anchor_weights(alg, c, p)
    term, masked, frac = anchor_ppo.AnchorPPO.anchor_terms(alg, mean, mu, CMDS)
    w = anchor_ppo.inrange_weights(CMDS)
    per = (torch.arange(n, dtype=torch.float32) ** 2)
    assert float(term) == pytest.approx(float((w * per).sum() / n))
    assert float(masked) == pytest.approx(float((w * per).sum() / w.sum()))
    assert frac == pytest.approx(float(w.mean()))
    alg_all = _alg("all")
    term_all, _, frac_all = anchor_ppo.AnchorPPO.anchor_terms(alg_all, mean, mu, CMDS)
    assert float(term_all) == pytest.approx(float(per.mean())) and frac_all == 1.0


def test_anchor_push_mask():
    alg = _alg("inrange", push=1.0)
    push_t = torch.full((CMDS.shape[0], 1), 5.0)
    push_t[0] = 0.5                                       # kicked half a second ago: no anchor
    w = anchor_ppo.AnchorPPO.anchor_weights(alg, CMDS, push_t)
    assert w[0] == 0 and w[1] == 1 and w.sum() == anchor_ppo.inrange_weights(CMDS).sum() - 1
    with pytest.raises(ValueError):
        anchor_ppo.AnchorPPO.anchor_weights(alg, CMDS, None)


def test_anchor_mlp_any_shape():
    sd = {"mlp.0.weight": torch.randn(16, 450), "mlp.0.bias": torch.randn(16), "mlp.2.weight": torch.randn(12, 16),
          "mlp.2.bias": torch.randn(12)}
    net = anchor_ppo._build_anchor_mlp(sd, "cpu")
    x = torch.randn(3, 450)
    ref = torch.nn.functional.elu(x @ sd["mlp.0.weight"].T + sd["mlp.0.bias"]) @ sd["mlp.2.weight"].T + sd["mlp.2.bias"]
    assert torch.allclose(net(x), ref, atol=1e-5)
    assert not any(p.requires_grad for p in net.parameters())


def test_mix_keys():
    for k, v in wtask.OMNI_MIXES.items():
        assert len(v) == 5 and math.isclose(sum(v), 1.0), k
    s = wtask.OMNI_MIXES["spin_up"]
    assert s[2] > wtask.OMNI_MIXES["default"][2]          # spins weighted up


# ------------------------------------------------------------------------------------------------ guards and options
def test_topple_start_and_lr_max():
    _, o = opts("--push-mps", "0", "0.6", "--push-interval", "4", "8", "--topple-mps", "0.6", "1.5",
                "--topple-interval", "8", "15", "--topple-start-iter", "300", step=100 * 24)
    assert o.topple_start_step == 400 * 24 and wtask.push_start_steps(o) == (0, 400 * 24)
    with pytest.raises(SystemExit):
        opts("--topple-start-iter", "300")                    # no tier 2
    a = tw.build_parser().parse_args(["--resume", "/x/m.pt", "--lr-max", "5e-5"])
    cfg = tw.runner_cfg(a, {})
    assert cfg["algorithm"]["lr_max"] == 5e-5 and cfg["algorithm"]["lr_min"] == 1e-5


def test_limits_contract_guard(monkeypatch):
    """The robot's limits record the calibration they were traced at (``contract_sha256_prefix``); training with the
    limits refuses another calibration unless ``--allow-limits-contract-mismatch``."""
    import hashlib

    from mp4_sim import limits as sim_limits

    from mp4_rl.common import bounds as limits
    from mp4_rl.common.policy import CONTRACT_JSON
    j = limits.load()
    have = hashlib.sha256(CONTRACT_JSON.read_bytes()).hexdigest()[:12]
    ok = dict(j, contract_sha256_prefix=have)
    rec = limits.limits_contract_check(False, ok)
    assert rec["match"] and rec["contract_in_force_sha256_prefix"] == have
    assert limits.limits_contract_check(False)["match"]              # the robot's own limits and calibration
    bad = dict(j, contract_sha256_prefix="0123456789ab")
    with pytest.raises(ValueError, match="allow the mismatch"):
        limits.limits_contract_check(False, bad)
    rec = limits.limits_contract_check(True, bad)
    assert not rec["match"] and rec["allow_limits_contract_mismatch"]
    # the trainer: refuses with --limits on a mismatch, passes with the flag, checks nothing without --limits
    monkeypatch.setattr(sim_limits, "load", lambda path=None, robot=None: bad)
    a = tw.build_parser().parse_args(["--resume", "/x/m.pt", "--limits"])
    with pytest.raises(SystemExit, match="allow-limits-contract-mismatch"):
        tw.limits_guard(a)
    a = tw.build_parser().parse_args(["--resume", "/x/m.pt", "--limits", "--allow-limits-contract-mismatch"])
    assert tw.limits_guard(a)["allow_limits_contract_mismatch"]
    assert tw.limits_guard(tw.build_parser().parse_args(["--resume", "/x/m.pt"])) is None


# ------------------------------------------------------------------------------------------------ fast commands and the heading hold
def test_fast_and_yaw_hold_flags():
    _, o = opts()
    assert (o.cmd_fast_share, o.yaw_zero_share, o.yaw_hold_weight) == (0.0, 0.0, 0.0)   # off: the plain draws
    _, o = opts("--cmd-fast-share", "0.4", "--cmd-fast-speed", "0.16", "0.2", "--cmd-fast-wz", "1.6", "2.0",
                "--yaw-zero-share", "0.5", "--yaw-hold-weight", "1.0", "--yaw-hold-std", "0.1")
    assert (o.cmd_fast_share, o.fast_speed, o.fast_wz) == (0.4, (0.16, 0.2), (1.6, 2.0))
    assert (o.yaw_zero_share, o.yaw_hold_weight, o.yaw_hold_std) == (0.5, 1.0, 0.1)
    for bad in (("--cmd-fast-share", "1.5"), ("--yaw-zero-share", "-0.1"), ("--cmd-fast-wz", "2", "1"),
                ("--yaw-hold-std", "0")):
        with pytest.raises(SystemExit):
            opts(*bad)


def test_yaw_hold_term():
    cmd = torch.tensor([[0.10, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.10, 0.03], [0.10, 0.0, 0.4], [0.0, 0.0, 0.0]])
    w = torch.tensor([0.0, 0.13, 0.03, 0.4, 0.0])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda name: cmd),
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(
                              root_link_ang_vel_b=torch.stack([w * 0, w * 0, w], -1)))})
    r = wtask.yaw_hold(env, std=0.10)
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(math.exp(-1.69), rel=1e-4)
    assert r[2] == pytest.approx(1.0)          # a loop-sized wz (<= 0.05) tracked exactly
    assert r[3] == 0 and r[4] == 0             # a turning walk and standing are not straight walks


@pytest.mark.parametrize("n", [2, 3])
def test_recipe_step2_and_step3_argv(tmp_path, n):
    a, o = recipe_argv(tmp_path, f"step{n}.sh", "71", "/abs/prev/model_7296.pt")
    assert a.resume == "/abs/prev/model_7296.pt"
    assert a.anchor == "mp4_rl/skills/walker/checkpoints/sku2/walker_start/model_6297.pt" and a.anchor_mask == "inrange"
    assert (a.iters, a.save_at, a.cmd_ramp, a.topple_start_iter) == ((1500, 1000)[n - 2], [500, 1000, 1500], 0, 0)
    assert o.spin_wz_max == (1.0, 2.0) and o.fwd_vx_range == (0.12, 0.20) and o.cmd_fast_share == 0.4
    assert o.yaw_hold_weight == 1.0 and o.spin_progress_weight == 0.0
    if n == 2:
        assert o.yaw_zero_share == 0.5 and o.fast_wz == (1.5, 2.0) and o.omni_shares is None
    else:
        assert o.yaw_zero_share == 0.3 and o.fast_wz == (1.0, 2.0)
        assert wtask.mix_shares(o) == (0.20, 0.25, 0.40, 0.05, 0.10)
    assert wtask.push_tiers(o) == ((0.0, 0.6, 4.0, 8.0), (0.6, 1.5, 8.0, 15.0))
    assert wtask.push_start_steps(o) == (0, 6297 * 24)
    assert (o.legs, o.actuator, o.r_scale_range) == ("sku_2", "md300", (0.95, 1.15))


# ------------------------------------------------------------------------------------------------ spin progress, turn logs
def _yaw_env(cmd, w, mode, step=0):
    term = SimpleNamespace(omni_mode=torch.tensor(mode))
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda name: cmd, get_term=lambda name: term),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.stack([w * 0, w * 0, w], -1)))},
        common_step_counter=step)


def test_spin_progress_term():
    cmd = torch.tensor([[0, 0, -2.0], [0, 0, -2.0], [0, 0, -2.0], [0, 0, 1.5], [0.1, 0, 0.8], [0.1, 0, 0.5], [0, 0, 0.2]])
    w = torch.tensor([-1.0, 0.0, 0.5, 2.0, 0.4, 0.5, 0.2])
    r = wtask.spin_progress(_yaw_env(cmd, w, [2, 2, 2, 2, 4, 1, 2]))
    assert r.tolist() == pytest.approx([0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0])   # partial pays; frozen, wrong way 0;
    # capped at 1; a plain omni walk's turn and |wz| <= 0.3 are not spins


def test_yaw_sign_stats_block():
    cmd = torch.tensor([[0, 0, -2.0], [0, 0, -1.0], [0, 0, 1.0], [0.1, 0, 0.0]])
    w = torch.tensor([-1.0, -1.0, 0.5, 0.0])
    env = _yaw_env(cmd, w, [2, 2, 2, 1], step=0)
    term = wtask.YawSignStats(None, env)
    for k in range(24):
        env.common_step_counter = k
        term(env)
    assert env.yaw_sign_stats == {}
    assert {k: float(v) for k, v in wtask.yaw_sign_log(env, None).items()} == {k: 0.0 for k in wtask.YAW_SIGN_KEYS}
    env.common_step_counter = 24
    term(env)
    s = env.yaw_sign_stats
    assert s["err_cw"] == pytest.approx(0.5) and s["ratio_cw"] == pytest.approx(0.75)
    assert s["err_ccw"] == pytest.approx(0.5) and s["ratio_ccw"] == pytest.approx(0.5)
    assert s["share_cw"] == pytest.approx(0.5) and s["share_ccw"] == pytest.approx(0.25)
    assert set(wtask.yaw_sign_log(env, None)) == set(s)


def test_recipe_step4_argv(tmp_path):
    a, o = recipe_argv(tmp_path, "step4.sh", "91", "/abs/prev/model_9794.pt")
    assert a.anchor == "mp4_rl/skills/walker/checkpoints/sku2/walker_start/model_6297.pt" and a.tag == "step4" and a.seed == 91
    assert (a.iters, a.save_at) == (1000, [500, 1000])
    assert wtask.mix_shares(o) == (0.20, 0.25, 0.40, 0.05, 0.10) and o.fast_wz == (1.0, 2.0)
    assert (o.cmd_fast_share, o.yaw_zero_share, o.yaw_hold_weight, o.spin_progress_weight) == (0.4, 0.3, 2.0, 1.0)
