#!/bin/sh
# Fine-tune the walker, step 2 of 4: continue from step 1 with the full command range from the start and the fast
# commands trained often, still held to the starting checkpoint's gait on the commands it knew.
#
#   sh skills/walker/recipes/sku2/step2.sh [SEED] [PARENT] [ITERS]
#
# SEED defaults to 81, PARENT to the last checkpoint of the newest run in runs/walker_step1 (a checkpoint with
# its run's config.json beside it), ITERS to 1500. Run from the repository root; output in runs/walker_step2. Environment:
# PYTHON (default .venv/bin/python), ENVS (robots, default 1024), OUT (the output folder), ANCHOR (default
# skills/walker/checkpoints/sku2/walker_start/model_6297.pt), EXTRA (more flags, appended last; argparse keeps the last value given).
#
# The settings, as step 1 (step1.sh) except:
# - commands: the full range (0.20 m/s, 2.0 rad/s turning in place) from the first iteration; 40 % of the walks and
#   spins drawn from the fast bands (0.15-0.20 m/s, 1.5-2.0 rad/s);
# - heading hold: half of the walks the heading loop does not steer are commanded with no turn, and a yaw-rate term
#   (weight 1.0, std 0.10 rad/s) pays for walking straight;
# - kicks of both kinds from the start;
# - model_<resumed + 500/1000/1500>.pt are kept besides every 50th.
set -eu
SEED=${1:-81}
PARENT=${2:-}
ITERS=${3:-1500}
if [ -z "$PARENT" ]; then
  run=$(ls -d runs/walker_step1/*_step1 2>/dev/null | tail -n 1)
  [ -n "$run" ] && PARENT=$(ls -v "$run"/model_*.pt | tail -n 1)
  [ -n "$PARENT" ] || { echo "no run in runs/walker_step1: give PARENT" >&2; exit 2; }
fi
ANCHOR=${ANCHOR:-skills/walker/checkpoints/sku2/walker_start/model_6297.pt}
# shellcheck disable=SC2086
exec env -u PYTHONPATH PYTHONPATH="$PWD/src" WANDB_MODE=disabled MUJOCO_GL=egl "${PYTHON:-.venv/bin/python}" -B \
  -m mp4_rl.walker.train \
  --resume "$PARENT" --anchor "$ANCHOR" --anchor-mask inrange --anchor-range 0.12 0.12 1.0 0.6 \
  --beta0 5 --beta-hold 100000 --beta-decay 0 --critic-warmup 30 --lr0 1e-4 \
  --terrain narrow --narrow-risers 0.025 0.045 --stair-exit --max-init-level 9 \
  --foot-softness 0.005 0.02 --friction 0.4 1.3 --tilt-limit 1.0 \
  --clearance apex --clearance-weight -2 --stumble -0.5 --progress 2 --overspeed -2 --omni-promote-rel 0.8 \
  --heading-law pi --heading-wz 0.4 --vx-max 0.12 \
  --omni-mix spin_up --cmd-full --cmd-ramp 0 --cmd-fast-share 0.4 \
  --yaw-zero-share 0.5 --yaw-hold-weight 1.0 --yaw-hold-std 0.10 \
  --push-mps 0 0.6 --push-interval 4 8 --topple-mps 0.6 1.5 --topple-interval 8 15 \
  --push-fall-grace 2 --anchor-push-mask 2 --save-at 500 1000 1500 \
  --limits --stand-offset-abd 12 --attachment-lidar 0.5 \
  --actuator md300 --supply-v 10.5 12.6 --r-scale 0.95 1.15 \
  --envs "${ENVS:-1024}" --iters "$ITERS" --seed "$SEED" --out "${OUT:-runs/walker_step2}" --tag step2 ${EXTRA:-}
