# mini_pupper_4_rl

Reinforcement-learning skills for the Mini Pupper v4: trained policies, the recipes that trained them, their evaluation
in simulation and the code that runs them on the robot. The robot models, their measured data and the simulation come
from [mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim) (package `mp4_sim`, release
v0.2.0).

## Skills

| Skill | What it does | SKU2 | SKU1 |
|---|---|---|---|
| [Walker](skills/walker/README.md) | Walks in every direction and turns in place, rides out shoves; in simulation also on rough ground and the printed stairs | policy, recipes, evaluation; ran on the robot | recipe and evaluation in simulation; no policy yet |
| [Get-up](skills/getup/README.md) | Gets up from any fall pose, with or without the lidar on the hood; switches with the walker on the robot | policy, recipes, evaluation; ran on the robot | recipe and evaluation in simulation; no policy yet |

## Robots

Every trainer, evaluator and the exporter take `--robot sku2|sku1` (default `sku2`; an evaluator defaults to the
policy's own robot).

- **SKU2:** the long body with a Raspberry Pi 5 and the 3S LiPo inside. The policies run on the Pi at 50 Hz and send
  servo targets over UART to the ESP32 board: [the robot guide](skills/getup/docs/ROBOT_SKU2.md).
- **SKU1:** the short body on the same legs, servos and board. Simulation only for now: running on the robot needs ESP32
  firmware that is not yet released ([ROBOT_SKU1.md](skills/getup/docs/ROBOT_SKU1.md)).

## Install

```bash
git clone https://github.com/mangdang-bam/mini_pupper_4_rl.git && cd mini_pupper_4_rl
bash scripts/install/install.sh        # installs mp4-sim v0.2.0 too; see docs/INSTALL.md
```

To install `mp4-sim` from elsewhere (a `mini_pupper_4_sim` checkout, or a pip URL such as
`git+file:///path/to/mini_pupper_4_sim@v0.2.0`), set `MP4_SIM_SOURCE` for the installer.

Training and evaluation need an NVIDIA GPU; without one the installer picks CPU mode (policy checks, export, the runner
on a simulated board, the CPU tests). The installer is for Linux and macOS; there is no Windows installer. Run the
commands below from the repository root, without an inherited `PYTHONPATH` (`env -u PYTHONPATH` if your shell sets
one).

Former name: mini_pupper_4_recovery (Python package `mp4_recovery`).

## Quick start

```bash
# If your shell sets PYTHONPATH, run each command with `env -u PYTHONPATH` in front.
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/cpu         # no GPU needed

# Check the shipped policies the way the robot does
.venv/bin/python -m mp4_rl.getup.runner check --policy skills/walker/policies/sku2
.venv/bin/python -m mp4_rl.getup.runner check --policy skills/getup/policies/sku2

# Evaluate in simulation (GPU): the get-up from every start, and walk, shove, fall, get up, walk for 60 s
sh skills/getup/eval/sku2.sh runs/eval gflat_s21 seq_s21
.venv/bin/python skills/getup/eval/score.py runs/eval

# Fine-tune the get-up from its starting checkpoint and export the result
sh skills/getup/recipes/sku2/finetune.sh
.venv/bin/python -m mp4_rl.common.export runs/getup_finetune/<run>/model_<k>.pt runs/exports/getup

# Put the policies on the Raspberry Pi (ubuntu@PI: the Pi's user and address): see skills/getup/docs/ROBOT_SKU2.md
tools/deploy.sh ubuntu@PI walk
```

On the robot one runner, `mp4_rl.getup.runner`, runs both policies: the walker, and the get-up when it detects a fall.

`tests/cpu/test_runner_modes.py` and the stream test in `tests/cpu/test_feedback.py` run robot-side code in real time
against a simulated board: on a busy machine a timing assertion can fail; rerun the failed test alone. `tests/cpu` skips
9 tests that need robot logs or an upstream checkout ([docs/INSTALL.md](docs/INSTALL.md#tests)).

## Results on the robot

Measured on one SKU2 with `skills/walker/policies/sku2` and `skills/getup/policies/sku2`, each take one run of the
runner; get-up times from the runner's log, speeds from the distance on a tape measure over the run's time, turn rates
from the gyro:

| | Takes | Result |
|---|---|---|
| Get-ups from placed fall poses: back, left side, right side, nose down; then back and right side with the lidar | one per pose: 4, and 2 with the lidar | every one up, standing in 1.5-1.7 s |
| Shoves by hand: standing (a 40 s run), and walking at 0.10 m/s (a 15 s run) | 2 runs, 6 shoves | every get-up the run's time allowed (5): up again in 1.4-1.7 s |
| Walking forward (0.10, 0.20 m/s), backward (0.10) and sideways (0.10, 0.20) | 7 | 0.69-0.91 × the speed simulation predicts for the same command (sideways the slowest), no falls |
| Turning in place at ±0.5, +1.0 and ±2.0 rad/s | 5 | 0.81-0.90 × the simulated turn rate; up to 1.71 rad/s and -1.83 rad/s at a ±2.0 rad/s command |
| Walking at 0.10 m/s while turning at 0.5 rad/s | 1 | about two thirds of the simulated turn (0.67 ×) |

No run logs are published. Simulation result tables are not published here either; each skill's README gives the
commands that produce them.

## What isn't supported yet

- SKU1 policies and running anything on the SKU1 (it needs ESP32 firmware that is not yet released).
- The printed stairs and rough ground are tested in simulation only, not on the robot.
- The runner's reading of the calibration from each policy folder has been tested on a simulated board, not yet on the
  robot ([the robot guide](skills/getup/docs/ROBOT_SKU2.md#known-limitations)).
- The shipped calibration and joint limits belong to one robot of each kind: another unit needs its own calibration
  ([the robot guide](skills/getup/docs/ROBOT_SKU2.md#adapting-to-another-unit)).

## What is where

```
skills/walker/     recipes/{sku2,sku1}/, policies/{sku2,sku1}/, checkpoints/sku2/ (start and final), eval/
skills/getup/      the same, and docs/ (running the policies on the robot)
src/mp4_rl/
  common/          the policy interface (numpy), decoder bounds, export, the base environment on mp4_sim's scene,
                   PPO with the smoothness losses and the anchor, the classic gait baseline
  walker/          the walker's task, training terrain, trainer and evaluations (omni, push, terrain)
  getup/           the get-up's task, trainer and evaluations (get-up, sequence); runner/ and mp4link/: the robot side,
                   numpy and pyserial only
tools/             deploy, stop, lifted playback, calibration and joint limits for another unit, run summaries,
                   robot-versus-simulation
scripts/install/   the installer and its checks
tests/             cpu/ needs no GPU, gpu/ 5-15 min
docs/              INSTALL, POLICY_CONTRACT
```
