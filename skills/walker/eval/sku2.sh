#!/bin/sh
# Evaluate the walker in simulation on the SKU2's plant, and the firmware's classic gait as its baseline: one JSON per
# evaluation in OUT, which skills/walker/eval/score.py turns into the result tables.
#
#   sh skills/walker/eval/sku2.sh OUT [NAME ...]
#
# NAME: run only these evaluations (e.g. omni_s21 push_s22); default all of them. An evaluation whose JSON is already in
# OUT is skipped, so the script can be run again after a stop. Each writes OUT/NAME.json and its output to OUT/NAME.log.
# A failed evaluation is reported and the others still run; the script then exits with status 1.
# Run from the repository root. Environment: PYTHON (default .venv/bin/python), WALKER (default
# skills/walker/policies/sku2), QUICK=1 runs each evaluation with an eighth of its robots (a quick check, not a result;
# give it its own OUT), VIDEOS=1 also records three short videos.
#
# The plant: the MD-300 servo model with a 3S supply (10.5-12.6 V), the weighed masses, the legs and joint limits from
# the policy's metadata. The lidar is a 35 mm high, 19 mm radius cylinder 43 mm forward of the base origin, 43.2 g.
# - the omni grid (the standard commands plus faster walks and turns; 24 robots per command), without and with the
#   lidar, seeds 21 and 22; kicks while standing and walking (push), seeds 21 and 22; rough ground (840 robots) and the
#   printed four-step stairs (256 robots), without and with the lidar, seed 21;
# - the classic gait: the same omni grid, its targets clamped like a policy's, seeds 21 and 22.
set -eu
[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 2; }
OUT=$1
shift
ONLY=" $* "
WALKER=${WALKER:-skills/walker/policies/sku2}
PY="env -u PYTHONPATH PYTHONPATH=$PWD/src WANDB_MODE=disabled MUJOCO_GL=egl ${PYTHON:-.venv/bin/python} -B"
PLANT="--actuator md300 --supply-v 10.5 12.6 --masses-json sku2_masses.json"
LIDAR="--attachment 35,19,0.043,43.2"
GRID="0.15,0,0;0.20,0,0;-0.10,0,0;-0.15,0,0;-0.20,0,0;0,0.15,0;0,0.20,0;0,-0.15,0;0,-0.20,0;0.1414,0.1414,0;\
-0.1414,-0.1414,0;0,0,0.5;0,0,-0.5;0,0,1.5;0,0,-1.5;0,0,2.0;0,0,-2.0;0.10,0,0.5;0.10,0,-0.5"
STAIRS="--terrain stairs_built --layout four_steps --width 0.40"
mkdir -p "$OUT"
FAILED=""

few() {  # FLAG N: with QUICK=1, FLAG and an eighth of N (the evaluation's robot count); otherwise nothing
  [ "${QUICK:-0}" = 1 ] || return 0
  echo "$1 $(( ($2 + 7) / 8 ))"
}

job() {  # NAME MODULE ARGS...: one evaluation, unless done or not asked for
  name=$1 mod=$2
  shift 2
  [ "$ONLY" = "  " ] || case "$ONLY" in *" $name "*) ;; *) return 0 ;; esac
  [ -f "$OUT/$name.json" ] && { echo "done $name"; return 0; }
  echo "run $name"
  # shellcheck disable=SC2086
  $PY -m "mp4_rl.walker.eval.$mod" "$@" --json "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || { echo "FAILED $name (see $OUT/$name.log)"; FAILED="$FAILED $name"; }
}

for s in 21 22; do
  job "omni_s$s" omni --policy "$WALKER" $PLANT --seed $s --per 24 $(few --per 24) --extra-commands "$GRID"
  job "omni_lidar_s$s" omni --policy "$WALKER" $PLANT $LIDAR --seed $s --per 24 $(few --per 24) --extra-commands "$GRID"
  job "push_s$s" push --policy "$WALKER" $PLANT --seed $s $(few --robots 64)
  job "classic_s$s" omni --classic --classic-clamp $PLANT --seed $s --per 24 $(few --per 24) --extra-commands "$GRID"
done
job wrough_s21 terrain --policy "$WALKER" $PLANT --terrain rough --envs 840 $(few --envs 840) --seed 21
job wrough_lidar_s21 terrain --policy "$WALKER" $PLANT $LIDAR --terrain rough --envs 840 $(few --envs 840) --seed 21
for t in "" _lidar; do
  [ -z "$t" ] && att="" || att=$LIDAR
  job "wstairs${t}_s21" terrain --policy "$WALKER" $PLANT $att $STAIRS --start-noise 0.01 2 \
    --envs 256 $(few --envs 256) --seconds 30 --heading-loop --heading-wz-max 0.4 --stop-at-goal --tilt-limit 1.0 \
    --seed 21
done
if [ "${VIDEOS:-0}" = 1 ]; then
  VC="0.10,0,0;0.20,0,0;0,0.10,0;0,0.20,0;0.1414,0.1414,0;0,0,1.0;0,0,2.0"
  job vid_side omni --policy "$WALKER" $PLANT --seed 21 --per 1 --video "$OUT/omni_side.mp4" \
    --video-commands "$VC" --video-seconds 4 --camera 0.5 -10 90
  job vid_front omni --policy "$WALKER" $PLANT --seed 21 --per 1 --video "$OUT/omni_front.mp4" \
    --video-commands "$VC" --video-seconds 4 --camera 0.5 -10 0
  job vid_stairs terrain --policy "$WALKER" $PLANT $STAIRS --envs 4 --seconds 20 --heading-loop --stop-at-goal \
    --tilt-limit 1.0 --seed 21 --video "$OUT/stairs.mp4"
fi
echo "results in $OUT: .venv/bin/python skills/walker/eval/score.py $OUT"
[ -z "$FAILED" ] || { echo "failed:$FAILED"; exit 1; }
