# Installing

## Requirements

- [uv](https://docs.astral.sh/uv/) 0.12.7 or newer (a release that knows the
  `cu130` torch backend). uv installs Python 3.12 if it is missing.
- GPU mode: an NVIDIA GPU with driver 580 or newer (torch is the CUDA 13.0
  build). The CUDA toolkit (`nvcc`) is unnecessary; torch ships its own CUDA
  runtime.
- CPU mode on macOS: macOS 14 or newer on Apple silicon. torch 2.14 has no
  wheel for Intel Macs.
- gcc, for the classic gait the walker's evaluations compare against
  (`mp4_rl/common/classic_gait.py` builds it on first use) and its tests.

## Modes

| Mode | For | Installs |
|---|---|---|
| `gpu` | Machines with an NVIDIA GPU | mjlab, MuJoCo Warp, warp-lang, rsl-rl-lib, CUDA 13.0 torch, plus everything in `cpu` |
| `cpu` | Any machine, including macOS and laptops without CUDA | mp4-sim, MuJoCo, numpy, scipy, pyserial, CPU torch, pytest |

Both modes install [mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim) (`mp4-sim`) at its
release v0.2.0: the robot models, their measured data (calibration, joint limits, masses), the servo models and the
simulation scenes this package trains and evaluates on. The Raspberry Pi installs nothing: `tools/deploy.sh` copies the
runner and the servo link, the policy interface, the SKU2 policy folders, three files of `mp4_sim` (`__init__.py`,
`calibration.py`, `imu_mount.json`), the first-step timing tool with the simulated board it uses, and any
`runs/streams/` (see the get-up's [robot guide](../mp4_rl/skills/getup/docs/ROBOT_SKU2.md#deploying)).

Training and evaluation need `gpu`. `cpu` runs the policy checks, the deployment
code and `tests/cpu`.

`auto` (the default) picks `gpu` when `nvidia-smi` is on `PATH`, and picks
`cpu` on macOS.

On Linux without an NVIDIA GPU, `--mode gpu --skip-verify` still installs the
`gpu` packages, for tools that simulate one robot on the CPU such as
`tools/getup_stream.py`; the check is skipped because it needs CUDA.

## Running the installer

```bash
bash scripts/install/install.sh [--mode auto|gpu|cpu] [--dry-run] [--skip-verify]
```

`MP4_SIM_SOURCE` installs `mp4-sim` from elsewhere: a folder (a `mini_pupper_4_sim` checkout) or a pip URL such as
`git+file:///path/to/mini_pupper_4_sim@v0.2.0`.

It:

1. Creates `.venv` in the project root with Python 3.12.
2. Installs `requirements/<mode>.txt` and `mp4-sim` under `requirements/constraints.txt`,
   choosing the torch build with uv's `--torch-backend` (`cu130` or `cpu`;
   on macOS, PyPI's torch).
3. Installs this package in editable mode. `pyproject.toml` lists numpy only
   (and the extras `pi` and `gpu`); backend packages and all pins come from
   `requirements/`. Each policy's folder, `mp4_rl/skills/<name>/`, is part of the
   package: its code with its recipes, exported policies, checkpoints and
   evaluation scripts. The recipes and evaluation scripts take paths relative to
   the repository root: run them from there.
4. In `gpu` mode, patches MuJoCo Warp (`.venv/bin/python -m mp4_sim.envs.patch_warp`,
   see below).
5. Checks (`scripts/install/verify.py`): `mujoco-core` is not installed (see
   "Pins and why"), every key package imports, torch sees
   CUDA (`gpu` mode), MuJoCo Warp is patched (`gpu` mode), `mp4_sim` is version
   0.2.0 (and, with `MP4_SIM_COMMIT` set, was installed from that commit), the
   robot's files match their recorded hashes, and each shipped policy's recorded
   calibration, joint limits and robot model hashes match `mp4_sim`'s files.

Use `--venv` for a different location. Re-running is safe; the venv is reused.

The package also installs as a regular (non-editable) package, for example into an environment of your own that already
has `mp4-sim`: `uv pip install .`, or a wheel built with `uv build`. The wheel holds the policy folders with their
policies and checkpoints (not the animations), so the runner's defaults and `scripts/install/verify.py` find them inside
the installed package.

## The MuJoCo Warp patch

When a geom touches more than 50 heightfield cells (a fallen body on fine
terrain), mujoco-warp 3.7.0.1 skips the extra contacts and prints a warning
from the GPU kernel, for every geom pair and every physics substep. With
thousands of robots on rough ground that slows a run several-fold and fills the
disk. `.venv/bin/python -m mp4_sim.envs.patch_warp` removes only the warning; the
contacts are handled as before. The feet never reach the limit. Reinstalling
mujoco-warp undoes the patch: run the command again. `--check` only reports.

## Pins and why

All versions live in `requirements/constraints.txt`: it pins every package the
installer installs, `mp4-sim`'s dependencies included, on every platform, so it
fixes the whole environment. It holds every pin of `mini_pupper_4_sim`'s own
constraints file at the same version, plus pyserial. The shipped policies were
trained and evaluated on this set, and the fitted servo models were identified
on the MuJoCo 3.7 line. Change a pin only together with a re-evaluation.

| Package | Version | Reason |
|---|---|---|
| Python | 3.12 | Required by mjlab and the actuator code |
| `mujoco` | 3.7.0 | The MuJoCo line the servo models were fitted on and the policies trained on |
| `mujoco-warp` | 3.7.0.1 | Matches `mujoco` 3.7.0; the version `patch_warp` patches |
| `warp-lang` | 1.12.1 | Version the policies were trained with; mjlab requires >= 1.12.0 |
| `mjlab` | 1.3.0 | Training framework |
| `rsl-rl-lib` | 5.0.1 | Pinned by mjlab 1.3.0; `common/algorithms.py` and `common/anchor_ppo.py` extend its PPO |
| `torch` | 2.14.0 | Version the policies were trained with; CUDA 13.0 wheels (`--torch-backend cu130`) |
| `mjviser` | 0.0.16 | mjlab's viewer, installed by mjlab. 0.0.17 and newer require `mujoco-core`, which installs its own `mujoco/` package over `mujoco` 3.7.0 and breaks `import mujoco` |
| `viser` | 1.1.1 | Required by `mjviser` 0.0.16 |

Every other pin (numpy, scipy, pyserial, tensordict, tensorboard, protobuf,
onnx, wandb and the packages they depend on; the nvidia-* and triton packages
only with CUDA torch on Linux) is the version the set resolves with.
`scripts/install/verify.py` fails if `mujoco-core` is installed; delete `.venv`
and run the installer again.

## Known risks

- **Headless Linux** needs EGL for rendering (videos from `mp4_rl.skills.getup.eval.getup_eval --video`):
  `export MUJOCO_GL=egl`.
- **Weights & Biases** is pulled in by mjlab. Training logs to TensorBoard; set
  `WANDB_MODE=disabled` when running offline.
- **Installing other packages into `.venv`** without `-c requirements/constraints.txt`
  can upgrade `mjviser` and bring in `mujoco-core` (see "Pins and why").

## Another unit's joint limits

The installer installs `mp4-sim` as a regular package, and its measured files and their recorded hashes are part of
it. To use joint limits traced on your own unit, install `mp4-sim` from a source checkout in editable mode and install
the limits into that checkout:

```bash
git clone --branch v0.2.0 https://github.com/mangdangroboticsclub/mini_pupper_4_sim.git ../mini_pupper_4_sim
uv pip install --python .venv/bin/python --no-deps -e ../mini_pupper_4_sim
.venv/bin/python tools/install_limits.py --limits my_sku2_joint_limits.json --hip-envelope my_sku2_hip_envelope.json
```

`tools/install_limits.py` checks the files, copies them over the robot's limits files in the checkout and records the
limits file's sha256 beside it. Then update the two hashes (`<robot>_joint_limits.json`, `<robot>_hip_envelope.json`)
in the checkout's `src/mp4_sim/robots.py`: until then `scripts/install/verify.py` reports the replaced files. Every
policy's decoder bounds come from the limits: re-export each policy
(`.venv/bin/python -m mp4_rl.common.export --from-policy DIR DIR`) and check a SKU2 policy with `runner check`
(the runner is the SKU2's). `--data FOLDER`
writes the files to another folder instead, without touching `mp4_sim`.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/cpu   # no GPU, CPU requirements only
.venv/bin/python -m pytest tests/gpu                           # needs the GPU requirements and at least 2 GB of GPU memory
```

Run the tests with the virtual environment's Python and without an inherited `PYTHONPATH` (`env -u PYTHONPATH`).

Two CPU tests run the robot-side code in real time against a simulated board and are sensitive to load:
`tests/cpu/test_runner_modes.py` (tick spacing) and `tests/cpu/test_feedback.py::test_stream_judges_the_track_end_on_filtered_readings`
(a stall of the stream loop on a busy machine trips the simulated board's watchdog). On a busy machine, such as one
training on the GPU, either can fail; rerun the failed test alone.

`tests/cpu` skips 9 tests unless two optional sources are named:

| Skipped | Needs | What it checks |
|---|---|---|
| `test_feedback.py::test_whole_robot_logs` (3) | `MP4_ROBOT_LOGS`: a folder with whole robot logs `map.npz`, `still60.npz`, `calib_a5.npz` (not part of this repository) | the feedback filter over whole runs recorded on the robot |
| `test_md300.py::test_torque_and_current_match_bam` (6) | `BAM_UPSTREAM`: a checkout of [Rhoban/bam](https://github.com/Rhoban/bam) with the MD-300 model (pull request #26, commit 2d1688a) | the vendored MD-300 model's torque and current against upstream's, three servos at 12.0 and 10.5 V |
