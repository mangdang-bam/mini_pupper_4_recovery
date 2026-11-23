#!/usr/bin/env bash
# Create the mp4-rl Python virtual environment (Linux / macOS), with the simulation package mp4-sim at its pinned
# release.
#
#   gpu mode - training and evaluation on an NVIDIA GPU, plus everything in cpu mode.
#   cpu mode - policy checks and export, the runner on a simulated board and the tests in tests/cpu.
#
# Usage:
#   bash scripts/install/install.sh [--mode auto|gpu|cpu] [--venv .venv]
#                                   [--python 3.12] [--dry-run] [--skip-verify]
#
# MP4_SIM_SOURCE: where mp4-sim comes from (default: its v0.2.0 release on GitHub): a folder (a mini_pupper_4_sim
# checkout), or a pip URL such as git+file:///path/to/mini_pupper_4_sim@v0.2.0. Headless Linux rendering: export
# MUJOCO_GL=egl.
set -euo pipefail

MODE="auto"
VENV_DIR=".venv"
PYTHON="3.12"
DRY_RUN=0
SKIP_VERIFY=0
SIM_DEFAULT="git+https://github.com/mangdangroboticsclub/mini_pupper_4_sim.git@v0.2.0"
SIM_SOURCE="${MP4_SIM_SOURCE:-$SIM_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2 ;;
    --venv)        VENV_DIR="$2"; shift 2 ;;
    --python)      PYTHON="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQ_DIR="$ROOT/requirements"
case "$VENV_DIR" in /*) VENV_PATH="$VENV_DIR" ;; *) VENV_PATH="$ROOT/$VENV_DIR" ;; esac
PY="$VENV_PATH/bin/python"

command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# a folder is installed from its files (not editable); anything else is a pip URL
if [[ -d "$SIM_SOURCE" ]]; then
  SIM_SPEC="mp4-sim @ file://$(cd "$SIM_SOURCE" && pwd)"
else
  SIM_SPEC="mp4-sim @ $SIM_SOURCE"
fi

OS="$(uname -s)"
if [[ "$MODE" == "auto" ]]; then
  if [[ "$OS" != "Darwin" ]] && command -v nvidia-smi >/dev/null 2>&1; then MODE=gpu; else MODE=cpu; fi
  echo "[mp4-rl] auto-detected mode: $MODE"
fi
if [[ "$OS" == "Darwin" && "$MODE" == "gpu" ]]; then
  echo "[mp4-rl] gpu mode is not supported on macOS (no CUDA)" >&2; exit 1
fi

# macOS torch wheels on PyPI are already CPU/MPS builds.
TORCH_FLAGS=()
if [[ "$OS" != "Darwin" ]]; then
  if [[ "$MODE" == "gpu" ]]; then TORCH_FLAGS=(--torch-backend=cu130); else TORCH_FLAGS=(--torch-backend=cpu); fi
fi

export PYTHONUTF8=1
export WANDB_MODE=disabled

echo "[mp4-rl] root    : $ROOT"
echo "[mp4-rl] venv    : $VENV_PATH"
echo "[mp4-rl] mode    : $MODE  (${TORCH_FLAGS[*]:-pypi torch})"
echo "[mp4-rl] mp4-sim : $SIM_SPEC"

uv venv --python "$PYTHON" --allow-existing "$VENV_PATH"

ARGS=(--python "$PY" -c "$REQ_DIR/constraints.txt" "${TORCH_FLAGS[@]}" -r "$REQ_DIR/$MODE.txt" "$SIM_SPEC" -e "$ROOT")
[[ $DRY_RUN -eq 1 ]] && ARGS+=(--dry-run)

echo "[mp4-rl] installing $MODE requirements ..."
uv pip install "${ARGS[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[mp4-rl] dry run complete - nothing installed."; exit 0
fi

if [[ "$MODE" == "gpu" ]]; then
  # mujoco_warp prints a warning per heightfield overflow from its GPU kernel; remove the warning (the physics is unchanged)
  echo "[mp4-rl] patching mujoco_warp ..."
  "$PY" -m mp4_sim.envs.patch_warp
fi

if [[ $SKIP_VERIFY -eq 0 ]]; then
  echo "[mp4-rl] verifying ..."
  "$PY" "$SCRIPT_DIR/verify.py" "$MODE"
fi

echo
echo "[mp4-rl] done. Activate with:  source $VENV_PATH/bin/activate"
