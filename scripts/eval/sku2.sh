#!/bin/sh
# Evaluate the shipped walker and get-up policy in simulation on the robot's plant, and the firmware's classic gait as the
# walker's baseline: one JSON per evaluation in OUT, which scripts/eval/score.py turns into the result tables.
#
#   sh scripts/eval/sku2.sh OUT [NAME ...]
#
# NAME: run only these evaluations (e.g. omni_s21 gflat_s22); default all of them, about 4 h on one GPU. An evaluation
# whose JSON is already in OUT is skipped, so the script can be run again after a stop. Each writes OUT/NAME.json and
# its output to OUT/NAME.log. Run from the repository root. Environment: PYTHON (default .venv/bin/python), WALKER
# (default policies/walker), GETUP (default policies/getup), VIDEOS=1 also records four short videos.
#
# The plant: the MD-300 servo model with a 3S supply (10.5-12.6 V), the weighed masses, the legs and joint limits from
# each policy's metadata. The lidar is a 35 mm high, 19 mm radius cylinder 43 mm forward of the base origin, 43.2 g.
# - walker: the omni grid (the standard commands plus faster walks and turns; 24 robots per command), without and with
#   the lidar, seeds 21 and 22; kicks while standing and walking (push_eval), seeds 21 and 22; rough ground (840
#   robots) and the printed four-step stairs (256 robots), without and with the lidar, seed 21;
# - classic gait: the same omni grid, its targets clamped like a policy's, seeds 21 and 22;
# - get-up: 512 robots from each start on flat ground, without and with the lidar, and the sequence (walk, kicks of
#   0.5-1.5 m/s every 2-4 s, fall, get up, walk; 256 robots for 60 s, with the walker), seeds 21-24; from every start on
#   rough ground and on the stairs (1024 robots), seeds 21 and 23.
set -eu
[ $# -ge 1 ] || { sed -n 2,23p "$0"; exit 2; }
OUT=$1
shift
ONLY=" $* "
WALKER=${WALKER:-policies/walker}
GETUP=${GETUP:-policies/getup}
PY="env -u PYTHONPATH PYTHONPATH=$PWD/src WANDB_MODE=disabled MUJOCO_GL=egl ${PYTHON:-.venv/bin/python} -B"
PLANT="--actuator md300 --supply-v 10.5 12.6 --masses-json sku2_masses.json"
LIDAR="--attachment 35,19,0.043,43.2"
GRID="0.15,0,0;0.20,0,0;-0.10,0,0;-0.15,0,0;-0.20,0,0;0,0.15,0;0,0.20,0;0,-0.15,0;0,-0.20,0;0.1414,0.1414,0;\
-0.1414,-0.1414,0;0,0,0.5;0,0,-0.5;0,0,1.5;0,0,-1.5;0,0,2.0;0,0,-2.0;0.10,0,0.5;0.10,0,-0.5"
STARTS="left,right,back,nose,midfall,random"
STAIRS="--terrain stairs_built --layout data/stairs/four_steps.json --width 0.40"
mkdir -p "$OUT"

job() {  # NAME MODULE ARGS...: one evaluation, unless done or not asked for
  name=$1 mod=$2
  shift 2
  [ "$ONLY" = "  " ] || case "$ONLY" in *" $name "*) ;; *) return 0 ;; esac
  [ -f "$OUT/$name.json" ] && { echo "done $name"; return 0; }
  echo "run $name"
  # shellcheck disable=SC2086
  $PY -m "mp4_recovery.eval.$mod" "$@" --json "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || echo "FAILED $name (see $OUT/$name.log)"
}

for s in 21 22; do
  job "omni_s$s" omni_eval --policy "$WALKER" $PLANT --seed $s --per 24 --extra-commands "$GRID"
  job "omni_lidar_s$s" omni_eval --policy "$WALKER" $PLANT $LIDAR --seed $s --per 24 --extra-commands "$GRID"
  job "push_s$s" push_eval --policy "$WALKER" $PLANT --seed $s
  job "classic_s$s" omni_eval --classic --classic-clamp $PLANT --seed $s --per 24 --extra-commands "$GRID"
done
job wrough_s21 terrain_eval --policy "$WALKER" $PLANT --terrain rough --envs 840 --seed 21
job wrough_lidar_s21 terrain_eval --policy "$WALKER" $PLANT $LIDAR --terrain rough --envs 840 --seed 21
for t in "" _lidar; do
  [ -z "$t" ] && att="" || att=$LIDAR
  job "wstairs${t}_s21" terrain_eval --policy "$WALKER" $PLANT $att $STAIRS --start-noise 0.01 2 --envs 256 \
    --seconds 30 --heading-loop --heading-wz-max 0.4 --stop-at-goal --tilt-limit 1.0 --seed 21
done
for s in 21 22 23 24; do
  job "gflat_s$s" getup_eval --policy "$GETUP" $PLANT --seed $s --robots 512 --seconds 8 --starts $STARTS
  job "glidar_s$s" getup_eval --policy "$GETUP" $PLANT $LIDAR --seed $s --robots 512 --seconds 8 --starts $STARTS
  job "seq_s$s" sequence --walker "$WALKER" --getup "$GETUP" $PLANT --schedule 0.5 1.5 2 4 --robots 256 --seconds 60 \
    --getup-phase-restart 1.0 --getup-timeout 20 --seed $s
done
for s in 21 23; do
  job "gterr_s$s" getup_eval --policy "$GETUP" $PLANT --seed $s --robots 1024 \
    --terrain low_poly,raised_relief,slope_up,slope_down,boxes --levels 3,6 --relief-pitch 0.02
  job "gstairs_s$s" getup_eval --policy "$GETUP" $PLANT --seed $s --robots 1024 --terrain stairs4,stairs8 --levels 3,6
done
if [ "${VIDEOS:-0}" = 1 ]; then
  VC="0.10,0,0;0.20,0,0;0,0.10,0;0,0.20,0;0.1414,0.1414,0;0,0,1.0;0,0,2.0"
  job vid_side omni_eval --policy "$WALKER" $PLANT --seed 21 --per 1 --video "$OUT/omni_side.mp4" \
    --video-commands "$VC" --video-seconds 4 --camera 0.5 -10 90
  job vid_front omni_eval --policy "$WALKER" $PLANT --seed 21 --per 1 --video "$OUT/omni_front.mp4" \
    --video-commands "$VC" --video-seconds 4 --camera 0.5 -10 0
  job vid_stairs terrain_eval --policy "$WALKER" $PLANT $STAIRS --envs 4 --seconds 20 --heading-loop --stop-at-goal \
    --tilt-limit 1.0 --seed 21 --video "$OUT/stairs.mp4"
  job vid_back getup_eval --policy "$GETUP" $PLANT --seed 21 --robots 4 --seconds 8 --starts back \
    --video "$OUT/getup_back.mp4" --video-start back
fi
echo "results in $OUT: python scripts/eval/score.py $OUT"
