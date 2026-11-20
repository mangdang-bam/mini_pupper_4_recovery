#!/bin/sh
# Fine-tune the walker for the SKU1: continue the SKU2's fine-tuned walker on the SKU1's simulated plant, with the
# settings of the SKU2's last step (skills/walker/recipes/sku2/step4.sh) except the plant and the anchor.
#
#   sh skills/walker/recipes/sku1/finetune.sh [SEED] [PARENT] [ITERS]
#
# SEED defaults to 201, PARENT to skills/walker/checkpoints/sku2/walker/model_10793.pt (the SKU2's fine-tuned walker, a
# checkpoint with its run's config.json beside it), ITERS to 1000. Run from the repository root; output in
# runs/walker_sku1. Environment: PYTHON (default .venv/bin/python), ENVS (robots, default 1024), OUT (the output
# folder), ANCHOR (a policy folder; default: the SKU2's walker exported for the SKU1, made by this script in
# OUT/anchor_sku1 if it is not there yet), EXTRA (more flags, appended last; argparse keeps the last value given).
#
# The anchor is the shipped SKU2 walker re-exported for the SKU1 (its decoder clamps from the SKU1's calibration and
# joint limits; the same network):
#
#   .venv/bin/python -m mp4_rl.common.export --from-policy skills/walker/policies/sku2 runs/walker_sku1/anchor_sku1 --robot sku1
#
# The settings, as step4.sh except:
# - the SKU1 (--robot sku1): its model, calibration, weighed masses (sku1_masses_413g.json) and its joint limits traced
#   on the robot, in walk mode;
# - the get-up's self-collision on the walker's plant (legs against legs, the hip cans and the body: the short body
#   puts the legs closer together);
# - the SKU1's lidar (35 mm high, radius 19 mm, 47.2 g with its mount) on half the robots, its axis drawn 20 mm behind
#   to 10 mm in front of the base origin and 10 mm either side;
# - the whole robot's centre of mass drawn 61-70 mm above the floor (the weighed robot: 65.9 mm).
set -eu
SEED=${1:-201}
PARENT=${2:-skills/walker/checkpoints/sku2/walker/model_10793.pt}
ITERS=${3:-1000}
OUT=${OUT:-runs/walker_sku1}
PY=${PYTHON:-.venv/bin/python}
if [ -z "${ANCHOR:-}" ]; then
  ANCHOR="$OUT/anchor_sku1"
  if [ ! -f "$ANCHOR/policy.npz" ]; then
    env -u PYTHONPATH PYTHONPATH="$PWD/src" "$PY" -B -m mp4_rl.common.export --from-policy skills/walker/policies/sku2 \
      "$ANCHOR" --robot sku1
  fi
fi
# shellcheck disable=SC2086
exec env -u PYTHONPATH PYTHONPATH="$PWD/src" WANDB_MODE=disabled MUJOCO_GL=egl "$PY" -B \
  -m mp4_rl.walker.train \
  --resume "$PARENT" --anchor "$ANCHOR" --anchor-mask inrange --anchor-range 0.12 0.12 1.0 0.6 \
  --beta0 5 --beta-hold 100000 --beta-decay 0 --critic-warmup 30 --lr0 1e-4 \
  --terrain narrow --narrow-risers 0.025 0.045 --stair-exit --max-init-level 9 \
  --foot-softness 0.005 0.02 --friction 0.4 1.3 --tilt-limit 1.0 \
  --clearance apex --clearance-weight -2 --stumble -0.5 --progress 2 --overspeed -2 --omni-promote-rel 0.8 \
  --heading-law pi --heading-wz 0.4 --vx-max 0.12 \
  --omni-mix spin_up --omni-shares 0.20 0.25 0.40 0.05 0.10 --cmd-full --cmd-ramp 0 \
  --cmd-fast-share 0.4 --cmd-fast-wz 1.0 2.0 --yaw-zero-share 0.3 --yaw-hold-weight 2.0 --yaw-hold-std 0.10 \
  --spin-progress-weight 1.0 \
  --push-mps 0 0.6 --push-interval 4 8 --topple-mps 0.6 1.5 --topple-interval 8 15 \
  --push-fall-grace 2 --anchor-push-mask 2 --save-at 500 1000 \
  --robot sku1 --masses-json sku1_masses_413g.json --limits --stand-offset-abd 12 \
  --leg-leg-contacts --attachment-lidar 0.5 --attachment-lidar-x -20 10 --attachment-lidar-y -10 10 \
  --com-height-mm 61 70 \
  --actuator md300 --supply-v 10.5 12.6 --r-scale 0.95 1.15 \
  --envs "${ENVS:-1024}" --iters "$ITERS" --seed "$SEED" --out "$OUT" --tag walker_sku1 ${EXTRA:-}
