#!/bin/sh
# Copy what the robot runs to its Raspberry Pi and check the policies there. The Pi needs python3, numpy and pyserial;
# nothing is installed.
#
#   tools/deploy.sh HOST FOLDER        HOST: an ssh host (user@pi); FOLDER: relative to the home directory, or absolute
#   tools/deploy.sh - FOLDER           copy into a local folder, to try the subset without a Pi
#
# What is copied, in the package's own layout: the runner and the servo link (mp4_rl/deploy/), the policy interface
# (mp4_rl/common/: policy, policy_file, heading), the walker's and the get-up's SKU2 policy folders (each with its
# calibration), mp4_sim's __init__.py and calibration.py and the robot's imu_mount.json (from the installed mp4_sim,
# located with PYTHON, default .venv/bin/python), the first-step timing tool and the simulated board it uses, and
# runs/streams if present. Then, on the Pi:
#
#   cd FOLDER
#   python3 -B -m mp4_rl.deploy.runner run --vx 0 --seconds 40 --log logs/run1
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
for f in __init__.py paths.py common/__init__.py common/policy.py common/policy_file.py common/heading.py; do
  copy "mp4_rl/$f" "mp4_rl/$f"
done
copy mp4_rl/deploy/ mp4_rl/deploy/
copy "$SIM/__init__.py" mp4_sim/__init__.py
copy "$SIM/calibration.py" mp4_sim/calibration.py
copy "$SIM/robot/assets/sku_2/measured/imu_mount.json" mp4_rl/deploy/runner/imu_mount.json
for skill in walker getup; do
  copy "mp4_rl/skills/$skill/policies/sku2/" "mp4_rl/skills/$skill/policies/sku2/"
done
copy tools/first_step_timing.py tools/first_step_timing.py
copy tests/cpu/fake_esp32.py tests/cpu/fake_esp32.py
[ -d runs/streams ] && copy runs/streams/ runs/streams/        # what tools/getup_stream.py writes by default
CHECK='for P in mp4_rl/skills/walker/policies/sku2 mp4_rl/skills/getup/policies/sku2; do python3 -B -m mp4_rl.deploy.runner check --policy $P || exit 1; done'
if [ "$HOST" = "-" ]; then
  mkdir -p "$DIR"
  rsync -a "$STAGE/" "$DIR/"
  echo "copied to $DIR"
else
  ssh -o ConnectTimeout=10 "$HOST" mkdir -p "$DIR"
  rsync -a "$STAGE/" "$HOST:$DIR/"
  ssh -o ConnectTimeout=10 "$HOST" "cd '$DIR' && $CHECK"
fi
