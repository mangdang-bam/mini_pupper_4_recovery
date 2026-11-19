#!/bin/sh
# Fine-tune the walker, step 3 of 4: continue from step 2 with turning weighted up.
#
#   sh scripts/recipes/walker_round3.sh [SEED] [PARENT] [ITERS]
#
# SEED defaults to 91, PARENT to the last checkpoint of the newest run in runs/walker_round2 (a checkpoint with
# its run's config.json beside it), ITERS to 1000. Run from the repository root; output in runs/walker_round3. Environment:
# PYTHON (default .venv/bin/python), ENVS (robots, default 1024), OUT (the output folder), ANCHOR (default
# checkpoints/walker_start/model_6297.pt), EXTRA (more flags, appended last; argparse keeps the last value given).
#
# The settings, as step 2 (walker_round2.sh) except:
# - the command shares (forward, sideways and diagonal walks, turning in place, standing, walking and turning):
#   0.20, 0.25, 0.40, 0.05, 0.10;
# - the fast band for turning in place 1.0-2.0 rad/s;
# - walks commanded with no turn: 30 % of those the heading loop does not steer.
set -eu
SEED=${1:-91}
PARENT=${2:-}
ITERS=${3:-1000}
if [ -z "$PARENT" ]; then
  run=$(ls -d runs/walker_round2/*_round2 2>/dev/null | tail -n 1)
  [ -n "$run" ] && PARENT=$(ls -v "$run"/model_*.pt | tail -n 1)
  [ -n "$PARENT" ] || { echo "no run in runs/walker_round2: give PARENT" >&2; exit 2; }
fi
ANCHOR=${ANCHOR:-checkpoints/walker_start/model_6297.pt}
# shellcheck disable=SC2086
exec env -u PYTHONPATH PYTHONPATH="$PWD/src" WANDB_MODE=disabled MUJOCO_GL=egl "${PYTHON:-.venv/bin/python}" -B \
  -m mp4_recovery.gpu.train_walker \
  --resume "$PARENT" --anchor "$ANCHOR" --anchor-mask inrange --anchor-range 0.12 0.12 1.0 0.6 \
  --beta0 5 --beta-hold 100000 --beta-decay 0 --critic-warmup 30 --lr0 1e-4 \
  --terrain narrow --narrow-risers 0.025 0.045 --stair-exit --max-init-level 9 \
  --foot-softness 0.005 0.02 --friction 0.4 1.3 --tilt-limit 1.0 \
  --clearance apex --clearance-weight -2 --stumble -0.5 --progress 2 --overspeed -2 --omni-promote-rel 0.8 \
  --heading-law pi --heading-wz 0.4 --vx-max 0.12 \
  --omni-mix spin_up --omni-shares 0.20 0.25 0.40 0.05 0.10 --cmd-full --cmd-ramp 0 --cmd-fast-share 0.4 \
  --cmd-fast-wz 1.0 2.0 --yaw-zero-share 0.3 --yaw-hold-weight 1.0 --yaw-hold-std 0.10 \
  --push-mps 0 0.6 --push-interval 4 8 --topple-mps 0.6 1.5 --topple-interval 8 15 \
  --push-fall-grace 2 --anchor-push-mask 2 --save-at 500 1000 1500 \
  --limits --stand-offset-abd 12 --attachment-lidar 0.5 \
  --actuator md300 --supply-v 10.5 12.6 --r-scale 0.95 1.15 \
  --envs "${ENVS:-1024}" --iters "$ITERS" --seed "$SEED" --out "${OUT:-runs/walker_round3}" --tag round3 ${EXTRA:-}
