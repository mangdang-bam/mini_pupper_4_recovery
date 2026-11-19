#!/usr/bin/env python
"""Train the get-up policy with PPO (``getup_task``). ``scripts/recipes/getup.sh`` is the training path: a behaviour
clone of the searched motions, then seven PPO steps of this script.

    python -m mp4_recovery.gpu.train --resume checkpoints/getup_start/model_6544.pt --limits --getup-phase 4 \
        --getup-band 90 180 180 --obs-norm --envs 2048 --iters 1000 --out runs/getup

``--resume`` fine-tunes a checkpoint: actor, critic, optimizer and iteration come from it; without it the networks start
fresh. The critic warms up first (``--critic-warmup``, value-only updates; ``algorithms.SmoothPPO``).

Network: ELU MLPs (``--actor-hidden``, ``--critic-hidden``). PPO: adaptive learning rate (KL 0.01; ``--lr0``,
``--lr-max``), gamma 0.99, lambda 0.95, clip 0.2, 5 epochs x 4 mini-batches, 24 steps per env per iteration.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper

from mp4_recovery import paths
from mp4_recovery.eval import built_env
from mp4_recovery.eval.built_env import masses_arg
from mp4_recovery.actuators import servo_model
from mp4_recovery.gpu.attachment import Attachment
from mp4_recovery.gpu.env_cfg import STEPS_PER_ITER, describe, noise_profile
from mp4_recovery.gpu.getup_task import (MIDFALL_OMEGA, GetupOptions, install_filter_ramp, make_getup_env_cfg,
                                         terrain_options)
from mp4_recovery.gpu.mdp import servo_action
from mp4_recovery.limits import add_legs_args, eval_legs, legs_flags
from mp4_recovery.obs_norm import freeze_updates
from mp4_recovery.paths import sha256


def runner_cfg(a) -> dict:
    cfg = asdict(RslRlOnPolicyRunnerCfg(
        seed=a.seed, num_steps_per_env=STEPS_PER_ITER, max_iterations=a.iters, save_interval=50,
        experiment_name="getup", logger="tensorboard", clip_actions=1.0,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        actor=RslRlModelCfg(hidden_dims=tuple(a.actor_hidden), activation="elu", obs_normalization=a.obs_norm,
                            distribution_cfg={"class_name": "mp4_recovery.gpu.getup_task:BoundedGaussianDistribution",
                                              "init_std": a.init_std, "std_type": "scalar",
                                              "std_min": a.std_bounds[0], "std_max": a.std_bounds[1]}),
        critic=RslRlModelCfg(hidden_dims=tuple(a.critic_hidden), activation="elu", obs_normalization=a.obs_norm),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=a.entropy_coef,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=a.lr0, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0, class_name="mp4_recovery.gpu.algorithms:SmoothPPO"),
    ))
    cfg["algorithm"].update(actor_warmup_iters=a.critic_warmup, caps_temporal=a.caps_temporal,
                            caps_spatial=a.caps_spatial, caps_obs_noise=a.caps_obs_noise, caps_seed=a.seed,
                            lr_min=1.0e-5, lr_max=a.lr_max)
    return cfg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--out", default=str(paths.RUNS / "getup"))
    ap.add_argument("--tag", default="", help="ends the run directory's name")
    ap.add_argument("--resume", help="a model_<k>.pt to fine-tune")
    # the task
    ap.add_argument("--limits", action="store_true", help="the measured joint limits, in the model and the decoder")
    ap.add_argument("--allow-limits-contract-mismatch", action="store_true",
                    help="with --limits: train even though the packaged limits were traced at another contract than "
                         "the one in force")
    add_legs_args(ap, "sku_2")
    ap.add_argument("--masses-json", type=masses_arg, metavar="PATH",
                    help="the plant's masses file (a path, or a file name in src/mp4_recovery/data/; default the task's, "
                         f"{GetupOptions.masses_json}); config.json records its path, sha256 and the built model's "
                         "total mass")
    ap.add_argument("--widen-limits-deg", type=float, metavar="DEG",
                    help="let hip and knee targets go DEG past their potentiometer track ends instead of stopping 20 "
                         "deg inside them. Training directly inside the limits never learns to flip from the back")
    ap.add_argument("--getup-band", type=float, nargs="+", metavar="DEG", default=(90.0, 90.0, 90.0),
                    help="the band about the stand (deg): abduction, hip, knee, or 12 values in servo order")
    ap.add_argument("--getup-phase", type=float, default=0.0, metavar="S",
                    help="the command floats carry (phi, sin 2 pi phi, cos 2 pi phi), phi = t / S")
    ap.add_argument("--getup-episode-s", type=float, default=6.0, help="the episode length (s)")
    ap.add_argument("--getup-mix", type=float, nargs=12, metavar="W",
                    help="the start mix (crouch partial side nose back random) at iteration 0, then from iteration "
                         "1000 on (normalised)")
    ap.add_argument("--getup-midfall", type=float, default=0.0, metavar="SHARE", help="the mid-fall starts' share")
    ap.add_argument("--getup-midfall-omega", type=float, nargs=2, default=MIDFALL_OMEGA, metavar=("LO", "HI"),
                    help="the mid-fall starts' rotation (rad/s)")
    ap.add_argument("--getup-end-on-success", action="store_true",
                    help="the episode ends when the success criterion has held 0.5 s")
    ap.add_argument("--getup-success-gyro", type=float, default=0.8,
                    help="the success reward's gyro limit (rad/s); 0: no gyro condition")
    ap.add_argument("--getup-action-rate", type=float, default=-0.02, help="the action-rate cost's weight")
    ap.add_argument("--getup-action-rate-tilted", type=float, default=0.1,
                    help="the action-rate cost's factor while tilted above 40 deg")
    ap.add_argument("--getup-joint-vel", type=float, default=0.0, help="weight (< 0) of the squared joint velocities")
    ap.add_argument("--getup-filter", type=float, default=0.0, metavar="ALPHA",
                    help="the target low-pass filter's final alpha; > 0 also moves the action-rate cost to the "
                         "applied target's change")
    ap.add_argument("--getup-filter-start", type=float, metavar="A0", help="the alpha at this run's first iteration")
    ap.add_argument("--getup-filter-ramp", type=int, default=0, metavar="N",
                    help="A0 -> ALPHA linearly over this run's first N iterations")
    ap.add_argument("--getup-terrain", action="store_true",
                    help="half the columns flat, half the rough families and the printed stairs, levels 0-7")
    ap.add_argument("--relief-pitch", type=float, metavar="M", help="--getup-terrain: the reliefs' heightfield pitch")
    ap.add_argument("--getup-terrain-stairs", type=int, default=1, metavar="N",
                    help="--getup-terrain: columns of each printed staircase")
    ap.add_argument("--getup-rollover-cost", type=float, default=0.0, metavar="W",
                    help="W >= 0: reward -W per second past 130 deg of tilt, in episodes that started below 100 deg")
    ap.add_argument("--getup-attachment", action="store_true",
                    help="a collision cylinder on the hood and its mass on the base, drawn per robot")
    ap.add_argument("--getup-attachment-height", type=float, nargs=2, default=Attachment.height_mm, metavar=("LO", "HI"),
                    help="how far it stands out of the hood (mm)")
    ap.add_argument("--getup-attachment-mass", type=float, nargs=2, default=Attachment.mass_g, metavar=("LO", "HI"),
                    help="its mass (g)")
    ap.add_argument("--getup-attachment-x", type=float, nargs=2, default=Attachment.x_mm, metavar=("LO", "HI"),
                    help="its axis, forward of the base origin (mm)")
    ap.add_argument("--getup-attachment-radius", type=float, default=Attachment.radius_mm, help="its radius (mm)")
    # the networks and PPO
    servo_model.add_args(ap)
    ap.add_argument("--actor-hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--obs-norm", action="store_true",
                    help="empirical observation normalisation in the actor and the critic (folded into the first "
                         "layer at export)")
    ap.add_argument("--obs-norm-freeze", type=int, default=0, metavar="N",
                    help="the actor's normaliser is not updated for this run's first N iterations")
    ap.add_argument("--init-std", type=float, default=0.1, help="the exploration std at the start")
    ap.add_argument("--std-bounds", type=float, nargs=2, default=(0.0, 1.0e9), metavar=("LO", "HI"),
                    help="clamp the action std to [LO, HI]")
    ap.add_argument("--entropy-coef", type=float, default=0.005, help="PPO's entropy bonus")
    ap.add_argument("--critic-warmup", type=int, default=0, help="value-only updates before the actor moves")
    ap.add_argument("--lr0", type=float, default=1.0e-3, help="the adaptive learning rate's start")
    ap.add_argument("--lr-max", type=float, default=1.0e-2, help="the adaptive learning rate's ceiling")
    ap.add_argument("--caps-temporal", type=float, default=0.0, metavar="LAMBDA", help="the temporal smoothness loss")
    ap.add_argument("--caps-spatial", type=float, default=0.0, metavar="LAMBDA", help="the spatial smoothness loss")
    ap.add_argument("--caps-obs-noise", type=float, default=0.0, metavar="SIGMA",
                    help="the spatial loss's perturbation, in the actor's normalised input units")
    a = ap.parse_args(argv)
    if a.getup_rollover_cost < 0.0:
        raise SystemExit("--getup-rollover-cost is a cost: W >= 0")
    return a


def task_options(a) -> GetupOptions:
    """The training task's options from the flags (``--masses-json`` absent: the task's default masses)."""
    mix = {}
    if a.getup_mix:
        start, end = a.getup_mix[:6], a.getup_mix[6:]
        mix = {"mix_start": tuple(w / sum(start) for w in start), "mix_end": tuple(w / sum(end) for w in end)}
    attachment = None
    if a.getup_attachment:
        attachment = Attachment(tuple(a.getup_attachment_height), tuple(a.getup_attachment_mass),
                                tuple(a.getup_attachment_x), a.getup_attachment_radius)
    opt = GetupOptions(num_envs=a.envs, noisy=True, noise_ramp=True, randomise=True, noise_end=noise_profile(),
                       band_deg=tuple(a.getup_band), success_gyro=a.getup_success_gyro or None, phase_s=a.getup_phase,
                       end_on_success=a.getup_end_on_success, midfall=a.getup_midfall, limits=a.limits,
                       widen_limits_deg=a.widen_limits_deg, episode_s=a.getup_episode_s,
                       action_rate_weight=a.getup_action_rate, action_rate_tilted_scale=a.getup_action_rate_tilted,
                       joint_vel_weight=a.getup_joint_vel, filter_alpha=a.getup_filter,
                       filter_alpha_start=a.getup_filter_start, filter_ramp_iters=a.getup_filter_ramp,
                       attachment=attachment, rollover_cost=a.getup_rollover_cost,
                       midfall_omega=tuple(a.getup_midfall_omega), **mix, **servo_model.options_from_args(a),
                       **eval_legs({}, **legs_flags(a)),
                       **({"masses_json": a.masses_json} if a.masses_json else {}),
                       **(terrain_options(a.relief_pitch, a.getup_terrain_stairs) if a.getup_terrain else {}))
    return opt


def main():
    a = parse_args()
    guard = None
    if a.limits:            # the packaged limits' calibration against the contract in force (limits_contract_check)
        from mp4_recovery.limits import limits_contract_check
        try:
            guard = limits_contract_check(a.allow_limits_contract_mismatch)
        except ValueError as e:
            raise SystemExit(str(e))
        print("limits contract", json.dumps(guard), flush=True)
    opt = task_options(a)
    env_cfg = make_getup_env_cfg(opt)
    env_cfg.seed = a.seed
    log_dir = Path(a.out) / (f"{datetime.now():%Y%m%d_%H%M%S}_s{a.seed}" + (f"_{a.tag}" if a.tag else ""))
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.terminations["nonfinite"].params["log_path"] = str(log_dir / "nonfinite.jsonl")

    cfg = runner_cfg(a)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    built = built_env.record(env, opt)
    print(built_env.line(built), flush=True)
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=1.0), cfg, str(log_dir), "cuda:0")
    norm_freeze = freeze_updates(runner.alg.actor, a.obs_norm_freeze) if a.obs_norm and a.obs_norm_freeze else None
    if a.resume:
        runner.load(a.resume, map_location="cuda:0",
                    load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True})

    commit = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(paths.REPO), "status", "--porcelain"], capture_output=True, text=True)
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "runner": cfg, "noise": "measured", "init_std": a.init_std, "critic_warmup": a.critic_warmup,
        "obs_norm": a.obs_norm, "obs_norm_freeze_iters": a.obs_norm_freeze if a.obs_norm else None,
        "resume": a.resume, "resume_sha256": sha256(Path(a.resume)) if a.resume else None,
        "masses_json": built["masses"], "built_env": {**built, "limits_contract": guard},
        "repo_commit": commit.stdout.strip(), "repo_dirty": bool(dirty.stdout.strip()), "argv": sys.argv},
        indent=1, default=str))
    if opt.filter_alpha or opt.filter_alpha_start:
        install_filter_ramp(runner, servo_action(env), opt.filter_alpha, opt.filter_alpha_start, opt.filter_ramp_iters)
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    if norm_freeze is not None:
        print("OBS_NORM_FREEZE", json.dumps(norm_freeze))
    print("RUN_DIR", log_dir)


if __name__ == "__main__":
    main()
