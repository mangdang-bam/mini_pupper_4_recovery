#!/bin/sh
# Evaluate the get-up in simulation on the SKU1's plant, alone and in sequence with the walker: one JSON per evaluation
# in OUT, which skills/getup/eval/score.py turns into the result tables.
#
#   sh skills/getup/eval/sku1.sh OUT [NAME ...]
#
# NAME: run only these evaluations (e.g. gflat_s21 seq_s21); default all of them. An evaluation whose JSON is already in
# OUT is skipped, so the script can be run again after a stop. Each writes OUT/NAME.json and its output to OUT/NAME.log.
# A failed evaluation is reported and the others still run; the script then exits with status 1.
# Run from the repository root. Environment: PYTHON (default .venv/bin/python), GETUP (default
# skills/getup/policies/sku1), WALKER (default skills/walker/policies/sku1, for the sequence; WALKER= skips the
# sequence), QUICK=1 runs each evaluation with an eighth of its robots (a quick check, not a result; give it
# its own OUT), VIDEOS=1 also records a short video.
#
# The plant: the MD-300 servo model with a 3S supply (10.5-12.6 V), the SKU1's weighed masses, the legs and joint limits from
# each policy's metadata. The lidar is a 35 mm high, 19 mm radius cylinder 5 mm behind the base origin, 47.2 g with its mount.
# - 512 robots from each start on flat ground, without and with the lidar, and the sequence (walk, kicks of 0.5-1.5 m/s
#   every 2-4 s, fall, get up, walk; 256 robots for 60 s, with the walker), seeds 21-24;
# - from every start on rough ground and on the stairs (1024 robots), seeds 21 and 23.
set -eu
[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 2; }
OUT=$1
shift
ONLY=" $* "
GETUP=${GETUP:-skills/getup/policies/sku1}
WALKER=${WALKER-skills/walker/policies/sku1}
PY="env -u PYTHONPATH PYTHONPATH=$PWD/src WANDB_MODE=disabled MUJOCO_GL=egl ${PYTHON:-.venv/bin/python} -B"
PLANT="--actuator md300 --supply-v 10.5 12.6 --masses-json sku1_masses_413g.json"
LIDAR="--attachment 35,19,-0.005,47.2"
STARTS="left,right,back,nose,midfall,random"
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
  $PY -m "mp4_rl.getup.eval.$mod" "$@" --json "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || { echo "FAILED $name (see $OUT/$name.log)"; FAILED="$FAILED $name"; }
}

for s in 21 22 23 24; do
  job "gflat_s$s" getup_eval --policy "$GETUP" $PLANT --seed $s --robots 512 $(few --robots 512) \
    --seconds 8 --starts $STARTS
  job "glidar_s$s" getup_eval --policy "$GETUP" $PLANT $LIDAR --seed $s --robots 512 $(few --robots 512) \
    --seconds 8 --starts $STARTS
  [ -z "$WALKER" ] || job "seq_s$s" sequence --walker "$WALKER" --getup "$GETUP" $PLANT --schedule 0.5 1.5 2 4 \
    --robots 256 $(few --robots 256) --seconds 60 --getup-phase-restart 1.0 --getup-timeout 20 --seed $s
done
for s in 21 23; do
  job "gterr_s$s" getup_eval --policy "$GETUP" $PLANT --seed $s --robots 1024 $(few --robots 1024) \
    --terrain low_poly,raised_relief,slope_up,slope_down,boxes --levels 3,6 --relief-pitch 0.02
  job "gstairs_s$s" getup_eval --policy "$GETUP" $PLANT --seed $s --robots 1024 $(few --robots 1024) \
    --terrain stairs4,stairs8 --levels 3,6
done
if [ "${VIDEOS:-0}" = 1 ]; then
  job vid_back getup_eval --policy "$GETUP" $PLANT --seed 21 --robots 4 --seconds 8 --starts back \
    --video "$OUT/getup_back.mp4" --video-start back
fi
echo "results in $OUT: .venv/bin/python skills/getup/eval/score.py $OUT"
[ -z "$FAILED" ] || { echo "failed:$FAILED"; exit 1; }
