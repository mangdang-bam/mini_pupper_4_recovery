#!/bin/sh
# Copy what the robot runs to its Raspberry Pi and check the policies there. The Pi needs python3, numpy and pyserial;
# nothing is installed.
#
#   tools/deploy.sh HOST FOLDER        HOST: an ssh host (user@pi); FOLDER: relative to the home directory, or absolute
#   tools/deploy.sh - FOLDER           copy into a local folder, to try the subset without a Pi
#
# What is copied: the runner and the servo link (src/mp4_rl: the get-up's runner/ and mp4link/, the policy interface),
# mp4_sim's __init__.py and calibration.py and the robot's imu_mount.json (from the installed mp4_sim, located with
# PYTHON, default .venv/bin/python), the walker's and the get-up's policy folders (each with its calibration), the
# first-step timing tool and the simulated board it uses, and runs/streams if present. Then, on the Pi:
#
#   cd FOLDER
#   R="env PYTHONPATH=src python3 -B -m mp4_rl.getup.runner"
#   $R run --vx 0 --seconds 40 --log logs/run1
set -eu
[ $# -eq 2 ] || { sed -n 2,15p "$0"; exit 2; }
HOST=$1
DIR=$2
cd "$(dirname "$0")/.."
SIM=$(${PYTHON:-.venv/bin/python} -c "import mp4_sim, pathlib; print(pathlib.Path(mp4_sim.__file__).resolve().parent)")
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
copy() {  # SRC DEST: one file or folder into the staging folder, without caches
  mkdir -p "$STAGE/$(dirname "$2")"
  rsync -a --exclude __pycache__ "$1" "$STAGE/$2"
}
for f in __init__.py paths.py common/__init__.py common/policy.py common/policy_file.py common/heading.py \
         getup/__init__.py; do
  copy "src/mp4_rl/$f" "src/mp4_rl/$f"
done
copy src/mp4_rl/getup/runner/ src/mp4_rl/getup/runner/
copy src/mp4_rl/getup/mp4link/ src/mp4_rl/getup/mp4link/
copy "$SIM/__init__.py" src/mp4_sim/__init__.py
copy "$SIM/calibration.py" src/mp4_sim/calibration.py
copy "$SIM/robot/assets/sku_2/measured/imu_mount.json" src/mp4_rl/getup/runner/imu_mount.json
copy skills/walker/policies/sku2/ skills/walker/policies/sku2/
copy skills/getup/policies/sku2/ skills/getup/policies/sku2/
copy tools/first_step_timing.py tools/first_step_timing.py
copy tests/cpu/fake_esp32.py tests/cpu/fake_esp32.py
[ -d runs/streams ] && copy runs/streams/ runs/streams/        # what tools/getup_stream.py writes by default
CHECK='for P in skills/walker/policies/sku2 skills/getup/policies/sku2; do env PYTHONPATH=src python3 -B -m mp4_rl.getup.runner check --policy $P || exit 1; done'
if [ "$HOST" = "-" ]; then
  mkdir -p "$DIR"
  rsync -a "$STAGE/" "$DIR/"
  echo "copied to $DIR"
else
  ssh -o ConnectTimeout=10 "$HOST" mkdir -p "$DIR"
  rsync -a "$STAGE/" "$HOST:$DIR/"
  ssh -o ConnectTimeout=10 "$HOST" "cd '$DIR' && $CHECK"
fi
