"""The classic controller (``classic_gait.py``): the firmware's gait and IK built on the host, its calibration mapping
onto our contract and model, and the tick -> target -> tick round trip; at the test calibration and the packaged one."""
import math
from pathlib import Path

import numpy as np
import pytest

from mp4_rl.common import classic_gait as cg
from mp4_rl.common.policy import CONTRACT_JSON, Contract

TEST_CONTRACT = Contract.load(Path(__file__).resolve().parent / "data" / "sku2_contract_test.json")
PACKAGED = Contract.load(CONTRACT_JSON)
# the firmware's own trot, tabulated at one robot's calibration (the npz's ``about``)
FIRMWARE_TROT = Path(__file__).resolve().parent / "data" / "firmware_trot_ticks.npz"


@pytest.mark.parametrize("c", [TEST_CONTRACT, PACKAGED], ids=["test", "packaged"])
@pytest.mark.parametrize("mul_first", [False, True])
def test_neutral_feet_are_the_contract_stand(c, mul_first):
    """The firmware's IK at its neutral stance (feet under the hips, 70 mm) gives servo degrees 0, so its ticks with
    the robot's offsets are the contract's stand ticks, and those encode the contract's stand dd."""
    g = cg.Gait(1, c.offsets_deg, mul_first=mul_first)
    sdeg, ticks = g.servo(cg.neutral_feet())
    assert np.all(sdeg == 0.0)
    assert np.array_equal(ticks[0], c.stand_ticks)
    assert np.array_equal(c.dd_from_ticks(ticks[0]), c.stand_dd)
    assert np.array_equal(c.ticks_from_servo_deg(np.zeros(12)), c.stand_ticks)


def test_gait_reset_is_the_stand_and_the_walk_starts_there():
    g = cg.Gait(1, PACKAGED.offsets_deg)
    feet = g.step(cg.firmware_command([[0.1, 0.0, 0.0]]))
    assert np.allclose(feet[0, :, 2], 70.0, atol=1e-4)   # the first tick: stance feet slide, z relaxed at 70
    assert np.allclose(feet[0, :, 1], 0.0)


@pytest.mark.parametrize("mul_first", [False, True])
def test_reproduces_the_firmware_trot_ticks(mul_first):
    """The firmware's own trot (its gait and IK built the same way), tabulated as 24 ticks of the steady cycle after 240
    ticks of warm-up at vx 50..200 mm/s and one calibration's offsets: every tick is reproduced from the vendored files."""
    d = np.load(FIRMWARE_TROT)
    for k, vx in enumerate(d["vx_mm_s"]):
        g = cg.Gait(1, d["offsets_deg"], mul_first=mul_first)
        for _ in range(240):
            g.step([vx, 0.0, 0.0])
        ticks = np.array([g.servo(g.step([vx, 0.0, 0.0]))[1][0] for _ in range(24)])
        assert np.array_equal(ticks, d["ticks"][k]), vx


def _leg_ik(x, y, z, leg):
    """pupper/Kinematics.py leg_explicit_inverse_kinematics in float64 (z negative below), written from the BSP."""
    L1, L2, off = 50.0, 60.0, (-26.0, 26.0, -26.0, 26.0)[leg]
    ryz = math.hypot(y, z)
    rh = math.sqrt(max(ryz ** 2 - 26.0 ** 2, 0.0))
    abd = math.acos(np.clip(off / ryz, -0.99, 0.99)) + math.atan2(z, y)
    theta = math.atan2(-x, rh)
    r = math.hypot(rh, x)
    hip = theta + math.acos(np.clip((L1 ** 2 + r ** 2 - L2 ** 2) / (2 * L1 * r), -0.99, 0.99))
    beta = math.acos(np.clip((L1 ** 2 + L2 ** 2 - r ** 2) / (2 * L1 * L2), -0.99, 0.99))
    return np.array([abd, hip, hip - (math.pi - beta)])


def _servo_deg_hand(feet, h0=70.0):
    sign = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, 1, -1]], dtype=float)   # SK_SIGN (axis x leg)
    out = np.zeros(12)
    for leg in range(4):
        sy = 1.0 if leg in (1, 3) else -1.0
        q = _leg_ik(feet[leg][0], sy * 26.0 + feet[leg][1], -feet[leg][2], leg)
        q0 = _leg_ik(0.0, sy * 26.0, -h0, leg)
        out[3 * leg:3 * leg + 3] = np.degrees(q - q0) * sign[:, leg]
    return out


def test_ik_matches_a_hand_computation():
    """The firmware's IK (float32) against the BSP's formulas written out here (float64): within 0.01 deg."""
    rng = np.random.default_rng(3)
    g = cg.Gait(1, np.zeros(12))
    for _ in range(50):
        feet = np.stack([rng.uniform(-40, 40, 4), rng.uniform(-25, 25, 4), rng.uniform(50, 90, 4)], -1)
        sdeg, _ = g.servo(feet[None].astype(np.float32))
        assert np.allclose(sdeg[0], _servo_deg_hand(feet.astype(np.float32).astype(float)), atol=0.01)


@pytest.mark.parametrize("c", [TEST_CONTRACT, PACKAGED], ids=["test", "packaged"])
def test_tick_target_tick_round_trip_is_exact(c):
    """Every tick of every servo -> the policy target ``rel_from_ticks`` (the tick's centre) -> the decoder's
    quantisation (numpy) -> the same tick and its dd; the torch decoder (float32, CPU) gives the same dd and target."""
    ticks = np.tile(np.arange(1024)[:, None], (1, 12))
    rel = c.rel_from_ticks(ticks)
    assert np.array_equal(c.ticks_from_servo_deg(c.servo_deg_from_rel(rel)), ticks)
    dd, phys = c.quantise_rel(rel)
    assert np.array_equal(dd, c.dd_from_ticks(ticks))
    torch = pytest.importorskip("torch")
    mdp = pytest.importorskip("mp4_rl.common.mdp")
    tc = mdp.TorchContract("cpu")
    tc.c = c
    tc.signs = torch.tensor(c.signs, dtype=torch.float32)
    tc.offsets = torch.tensor(c.offsets_deg, dtype=torch.float32)
    dd_t, phys_t = tc.quantise_rel(torch.tensor(rel.astype(np.float32)))
    assert np.array_equal(dd_t.numpy().astype(np.int64), dd)
    assert np.allclose(phys_t.numpy(), phys, atol=1e-6)


def test_formula_difference_is_rare():
    """firmware-dev's deg * (10240 / 3100) against the contract's (deg * 10240) / 3100 over an omni grid of steady
    cycles: the ticks differ (by one) in at most 0.5 % of the entries."""
    n = d = 0
    for vx in (-200, -100, 0, 100, 200):
        for vy in (-200, 0, 200):
            for wz in (-2.0, 0.0, 2.0):
                a, b = cg.Gait(1, PACKAGED.offsets_deg), cg.Gait(1, PACKAGED.offsets_deg, mul_first=True)
                for _ in range(72):
                    f = a.step([vx, vy, wz])
                    b.step([vx, vy, wz])
                    ta, tb = a.servo(f)[1], b.servo(f)[1]
                    n += ta.size
                    d += int((ta != tb).sum())
                    assert np.abs(ta - tb).max() <= 1
    assert d / n <= 0.005, (d, n)


def test_batch_equals_single_robots():
    """Robots stepped together, with masks and different commands, equal each robot stepped alone."""
    cmds = np.array([[100, 0, 0], [0, 150, 0], [0, 0, 1.5], [200, -200, -2.0]], dtype=np.float32)
    g = cg.Gait(4, PACKAGED.offsets_deg)
    mask = np.array([1, 1, 0, 1])
    out = []
    for k in range(60):
        m = mask if k < 10 else None
        out.append(g.servo(g.step(cmds, m))[1])
    for r in range(4):
        s = cg.Gait(1, PACKAGED.offsets_deg)
        for k in range(60):
            if k < 10 and not mask[r]:
                continue
            t = s.servo(s.step(cmds[r:r + 1]))[1][0]
            assert np.array_equal(t, out[k][r]), (r, k)


def test_trot_timing():
    """A 24-tick (360 ms) cycle; each foot lifted (>= 5 mm) on 5 ticks per cycle, at most the 20 mm clearance; the
    diagonal pairs (FR, RL) and (FL, RR) swing together and the two pairs never together."""
    g = cg.Gait(1, PACKAGED.offsets_deg)
    z = np.array([g.step([100.0, 0.0, 0.0])[0, :, 2] for _ in range(96)])[48:]
    up = z < 70.0 - 5.0
    assert np.array_equal(up[:24], up[24:48])
    assert list(up[:24].sum(0)) == [5, 5, 5, 5]
    assert abs(z.min() - 50.0) < 1e-3
    assert np.array_equal(up[:, 0], up[:, 3]) and np.array_equal(up[:, 1], up[:, 2])
    assert not np.any(up[:, 0] & up[:, 1])


def test_firmware_command_clamp_and_start():
    c = cg.firmware_command([[0.5, -0.3, 3.0], [0.1, 0.0, 0.0], [0.001, 0.0, 0.01]])
    assert np.allclose(c[0], [200.0, -200.0, 2.0])
    assert np.allclose(c[1], [100.0, 0.0, 0.0], atol=1e-4)
    assert list(cg.moving(c)) == [True, True, False]


def _model_feet(c, legs, rel):
    import mujoco
    from mp4_sim.robot import model as sku2
    m = sku2.get_spec(legs=legs).compile()
    d = sku2._stand_data(m)
    model = (c.stand_rel_rad + rel).reshape(4, 3)
    for li, L in enumerate(sku2.LEGS):
        kt, kc = c.meta["tendon_coefs"][L][f"{L}_thigh"], c.meta["tendon_coefs"][L][f"{L}_calf"]
        d.qpos[m.jnt_qposadr[m.joint(f"{L}_hip").id]] = model[li, 0]
        d.qpos[m.jnt_qposadr[m.joint(f"{L}_thigh").id]] = model[li, 1]
        d.qpos[m.jnt_qposadr[m.joint(f"{L}_calf").id]] = (model[li, 2] - kt * model[li, 1]) / kc
    mujoco.mj_forward(m, d)
    return np.array([d.site_xpos[m.site(f"{L}_foot").id] for L in sku2.LEGS])


@pytest.mark.parametrize("foot", ["sphere", "mesh"])
def test_calibration_mapping_moves_the_model_feet_as_commanded(foot):
    """The mapping firmware foot -> servo degrees -> contract rel (``sign * radians``) -> the model's joints (abduction
    and thigh angles, the knee tendon): a 10 mm foot move forward, left or up in the firmware's frame moves every
    model foot site 9-10.5 mm that way (the model's links are not the IK's 50/60 mm), less than 2 mm across."""
    pytest.importorskip("mujoco")
    from mp4_sim.robot import model as sku2
    lg = sku2.Legs(foot=foot)
    c = PACKAGED
    g = cg.Gait(1, c.offsets_deg)
    f0 = _model_feet(c, lg, np.zeros(12))
    for dv, axis in (((10, 0, 0), np.array([1.0, 0, 0])), ((0, 10, 0), np.array([0, 1.0, 0])),
                     ((0, 0, -10), np.array([0, 0, 1.0]))):
        feet = cg.neutral_feet()
        feet[0] += np.array(dv, dtype=np.float32)
        sdeg, _ = g.servo(feet)
        dp = 1000.0 * (_model_feet(c, lg, c.rel_from_servo_deg(sdeg[0])) - f0)
        along = dp @ axis
        across = np.linalg.norm(dp - along[:, None] * axis, axis=-1)
        assert np.all((along > 9.0) & (along < 10.5)), (foot, dv, dp)
        assert np.all(across < 2.0), (foot, dv, dp)
