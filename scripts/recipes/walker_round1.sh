#!/bin/sh
# Fine-tune the walker, step 1 of 4: resume the walker's starting checkpoint on this robot's simulated plant, held to its
# own gait on the commands it already knew while the command ranges widen to 0.20 m/s and 2.0 rad/s.
#
#   sh scripts/recipes/walker_round1.sh [SEED] [PARENT] [ITERS]
#
# SEED defaults to 71, PARENT to checkpoints/walker_start/model_6297.pt (a checkpoint with its run's config.json beside
# it), ITERS to 1000. Run from the repository root; output in runs/walker_round1. Environment: PYTHON (default
# .venv/bin/python), ENVS (robots, default 1024), OUT (the output folder), EXTRA (more flags, appended last; argparse
# keeps the last value given).
#
# The settings:
# - anchor: the starting checkpoint, beta 5 for the whole run, only on commands inside its trained range (0.12 m/s,
#   1.0 rad/s turning in place, 0.6 rad/s while walking) and not within 2 s of a kick;
# - commands: widened from that range to 0.20 m/s forward, sideways and backward, 2.0 rad/s turning in place and
#   1.0 rad/s while walking, over the first 300 iterations; more turning in place than the default command mix
#   (`--omni-mix spin_up`: forward 25 %, other walks 30 %, turning in place 25 %, standing 10 %, walking and turning
#   10 %);
# - terrain and rewards: narrow flights with 25-45 mm risers (an episode on stairs ends as a success once the robot is off
#   them), foot clearance, stumble, progress and overspeed terms, the heading loop;
# - plant: the MD-300 servo model with a 3S supply (10.5-12.6 V) and winding resistance x0.95-1.15, the weighed masses
#   (the package data file sku2_masses.json, in src/mp4_recovery/data/), the measured joint limits in walk mode, the
#   abductions' stand 12 deg wider, the lidar on half the robots;
# - kicks: 0-0.6 m/s every 4-8 s from the start, 0.6-1.5 m/s every 8-15 s from iteration 300;
# - model_<resumed + 300>.pt is kept besides every 50th.
set -eu
SEED=${1:-71}
PARENT=${2:-checkpoints/walker_start/model_6297.pt}
ITERS=${3:-1000}
# shellcheck disable=SC2086
exec env -u PYTHONPATH PYTHONPATH="$PWD/src" WANDB_MODE=disabled MUJOCO_GL=egl "${PYTHON:-.venv/bin/python}" -B \
  -m mp4_recovery.gpu.train_walker \
  --resume "$PARENT" --anchor "$PARENT" --anchor-mask inrange --anchor-range 0.12 0.12 1.0 0.6 \
  --beta0 5 --beta-hold 100000 --beta-decay 0 --critic-warmup 30 --lr0 1e-4 \
  --terrain narrow --narrow-risers 0.025 0.045 --stair-exit --max-init-level 9 \
  --foot-softness 0.005 0.02 --friction 0.4 1.3 --tilt-limit 1.0 \
  --clearance apex --clearance-weight -2 --stumble -0.5 --progress 2 --overspeed -2 --omni-promote-rel 0.8 \
  --heading-law pi --heading-wz 0.4 --vx-max 0.12 \
  --omni-mix spin_up --cmd-full --cmd-ramp 300 \
  --push-mps 0 0.6 --push-interval 4 8 --topple-mps 0.6 1.5 --topple-interval 8 15 --topple-start-iter 300 \
  --push-fall-grace 2 --anchor-push-mask 2 --save-at 300 \
  --limits --stand-offset-abd 12 --attachment-lidar 0.5 \
  --actuator md300 --supply-v 10.5 12.6 --r-scale 0.95 1.15 \
  --envs "${ENVS:-1024}" --iters "$ITERS" --seed "$SEED" --out "${OUT:-runs/walker_round1}" --tag round1 ${EXTRA:-}
