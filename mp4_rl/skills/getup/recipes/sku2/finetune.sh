#!/bin/sh
# Fine-tune the get-up policy on this robot's simulated plant: resume the get-up's starting checkpoint with uneven
# ground, printed stairs and a random load on the back added.
#
#   sh mp4_rl/skills/getup/recipes/sku2/finetune.sh [SEED] [PARENT] [ITERS]
#
# SEED defaults to 72, PARENT to mp4_rl/skills/getup/checkpoints/sku2/getup_start/model_6544.pt, ITERS to 1000. Run from the repository root;
# output in runs/getup_finetune. Environment: PYTHON (default .venv/bin/python), ENVS (robots, default 1536), OUT (the
# output folder), EXTRA (more flags, appended last; argparse keeps the last value given).
#
# The settings:
# - starts: crouched, partly fallen, on a side, nose down, on the back and with random joints (5/5/25/15/40/10 %),
#   a fifth of the episodes from a mid-fall; 8 s episodes, a 4 s phase clock, an episode ends once the robot is up;
# - the action band 90/180/180 deg (abduction, hip, knee) inside the measured joint limits, a target filter of 0.75;
# - costs on the action rate and joint speed, the smoothness loss (`--caps-*`: temporal 2, spatial 40, observation noise 0.05);
# - terrain: half the columns flat, half rough ground (relief on a 2 cm grid, slopes, boxes) and the printed four- and
#   eight-step stairs (four columns each);
# - a cylinder on the hood, drawn per robot (0-40 mm high, 0-70 g), standing in for the lidar and other add-ons;
# - plant: the MD-300 servo model with a 3S supply (10.5-12.6 V) and winding resistance x0.95-1.15, the weighed masses
#   (sku2_masses.json, the robot's measured data in mp4_sim);
# - PPO: learning rate 1e-5 to at most 5e-5, the critic warmed up for 30 iterations; the policy's std is the checkpoint's
#   own (0.03; --init-std only sets a fresh actor's), clamped to 0.06-0.3 when sampling.
set -eu
SEED=${1:-72}
PARENT=${2:-mp4_rl/skills/getup/checkpoints/sku2/getup_start/model_6544.pt}
ITERS=${3:-1000}
# shellcheck disable=SC2086
exec env -u PYTHONPATH PYTHONPATH="$PWD" WANDB_MODE=disabled MUJOCO_GL=egl "${PYTHON:-.venv/bin/python}" -B \
  -m mp4_rl.skills.getup.train \
  --resume "$PARENT" --limits --getup-phase 4 --getup-band 90 180 180 --getup-episode-s 8 \
  --init-std 0.03 --entropy-coef 0 --getup-success-gyro 0 --getup-end-on-success --getup-midfall 0.2 \
  --getup-mix 0.05 0.05 0.25 0.15 0.40 0.10 0.05 0.05 0.25 0.15 0.40 0.10 \
  --actor-hidden 512 256 128 --critic-hidden 512 256 128 --obs-norm --obs-norm-freeze 100 \
  --lr0 1e-5 --lr-max 5e-5 --critic-warmup 30 \
  --getup-action-rate=-0.1 --getup-action-rate-tilted 1.0 --getup-joint-vel=-5e-4 \
  --getup-filter 0.75 --getup-filter-start 0.75 --getup-filter-ramp 0 \
  --caps-temporal 2 --caps-spatial 40 --caps-obs-noise 0.05 \
  --getup-terrain --relief-pitch 0.02 --getup-terrain-stairs 4 --getup-attachment --std-bounds 0.06 0.3 \
  --actuator md300 --supply-v 10.5 12.6 --r-scale 0.95 1.15 \
  --envs "${ENVS:-1536}" --iters "$ITERS" --seed "$SEED" --out "${OUT:-runs/getup_finetune}" --tag getup ${EXTRA:-}
