# The walker

A walking policy for the Mini Pupper v4: walks in every direction, turns in place and rides out shoves; in simulation
also on rough ground and the printed stairs. It runs on its own, or with the [get-up](../getup/README.md), which takes
over after a fall.

| Robot | Policy | Inputs | Status |
|---|---|---|---|
| SKU2 | [`policies/sku2`](policies/sku2) | 450 (10 frames × 45) | ran on the robot |
| SKU1 | [`policies/sku1`](policies/sku1) | 450 (10 frames × 45) | ran on the robot |

The policy folder holds `policy.npz` (the network and its metadata, [POLICY_CONTRACT.md](../../docs/POLICY_CONTRACT.md)),
`config.json` (the training run's options) and copies of the calibration and joint limits it was exported for.

## How it was made

Fine-tuned in four steps on the SKU2's simulated plant (the simulated robot as training and evaluation see it) from
`checkpoints/sku2/walker_start/model_6297.pt`. The plant: the SKU2 model with its measured joint limits and weighed masses (630.5 g), the MD-300 servo model on a 3S supply
(10.5-12.6 V) with the winding resistance drawn from 0.95-1.15 × its fitted value; all from
[mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim). Training needs the `gpu` install
([INSTALL.md](../../docs/INSTALL.md)). Run the recipes from the repository root; each script's header lists every
setting.

```sh
sh skills/walker/recipes/sku2/step1.sh      # from checkpoints/sku2/walker_start/model_6297.pt
sh skills/walker/recipes/sku2/step2.sh      # from the last checkpoint of the newest step-1 run, and so on to step4.sh
```

Each script takes `[SEED] [PARENT] [ITERS]`, and `ENVS`, `OUT` and `EXTRA` from the environment; its output goes to
`runs/walker_step<N>/`.

| Recipe | Parent | Seed | Iterations | What it changes |
|---|---|---|---|---|
| `step1.sh` | `walker_start/model_6297.pt` | 71 | 1000 | Widens the commands over the first 300 iterations, from 0.12 m/s and 1.0 rad/s to 0.20 m/s and 2.0 rad/s turning in place; kicks up to 1.5 m/s from iteration 300 |
| `step2.sh` | step 1's last checkpoint | 81 | 1500 | The full command range from the start, 40 % of the walks and turns drawn from the fast end, and a reward for holding the heading |
| `step3.sh` | step 2's last checkpoint | 91 | 1000 | Turning in place weighted up to 40 % of the commands, fast turns drawn from 1.0-2.0 rad/s |
| `step4.sh` | step 3's last checkpoint | 101 | 1000 | Pays for partial turns, so fast turns in either direction keep turning; a stronger heading hold |

`checkpoints/sku2/walker/model_10793.pt` is step 4's last checkpoint: `policies/sku2` was exported from it. On an RTX 5080
(16 GB) running three runs at a time, 1000 iterations take about 2-3 h. These are the settings used for the shipped
policy, not a promise of the same numbers: a run's result depends on its seed and the machine.

**A quick check of a recipe** (fewer robots and a few iterations, in its own output folder). The positional arguments
are `[SEED] [PARENT] [ITERS]`; an empty one keeps its default:

```sh
ENVS=64 OUT=runs/quick sh skills/walker/recipes/sku2/step1.sh "" "" 5
```

A run saves `model_<k>.pt` where k counts on from the parent's iteration: every 50 iterations (k a multiple of 50), at
300 iterations after the parent and at the end, k = the parent's iteration + ITERS - 1 (`model_6301.pt` from
`model_6297.pt` with 5 iterations). Such a checkpoint exports and passes `runner check` like any other; it is not a
trained policy. The next step looks for its parent in `runs/walker_step<N>/`, so a run in another `OUT` doesn't become
one.

### The SKU1

`recipes/sku1/finetune.sh` continues the SKU2's fine-tuned walker on the SKU1's simulated plant with step 4's settings:

| Recipe | Parent | Seed | Iterations |
|---|---|---|---|
| `sku1/finetune.sh` | `checkpoints/sku2/walker/model_10793.pt` | 210 | 1000 |

What it changes:

- the SKU1 (`--robot sku1`): its model, calibration, joint limits traced on the robot and weighed masses (413.4 g);
- rough ground (relief, slopes, steps and boxes) instead of the narrow stairs;
- the get-up's self-collision on the walker's plant;
- the SKU1's lidar (47.2 g with its mount, its axis drawn 20 mm behind to 10 mm in front of the base origin and 10 mm
  either side) on half the robots;
- the whole robot's centre of mass drawn 61-70 mm above the floor;
- anchored (below) to the shipped SKU2 walker exported for the SKU1, over the whole command range (up to 0.20 m/s and
  2.0 rad/s turning in place, 1.0 rad/s while walking) and with weight 10 instead of 5.

`checkpoints/sku1/walker/model_11792.pt` is its last checkpoint: `policies/sku1` was exported from it. The run that made
it used the walk-mode joint limits estimated before the SKU1's legs were traced (the SKU2's measured ranges at the
SKU1's calibration); the recipe uses the traced limits, which the policy is exported for. Other seeds of the recipe came
out weaker in simulation than this run: evaluate a new run before using it.

The anchor is a policy folder the script makes before training, unless `ANCHOR` names another:

```sh
.venv/bin/python -m mp4_rl.common.export --from-policy skills/walker/policies/sku2 runs/walker_sku1/anchor_sku1 \
    --robot sku1
```

The export keeps the network and takes the decoder's clamps from the SKU1's calibration and joint limits. The script
takes `[SEED] [PARENT] [ITERS]` and `ENVS`, `OUT`, `ANCHOR`, `EXTRA` from the environment; output in `runs/walker_sku1/`.

### The trainer

`python -m mp4_rl.walker.train` resumes a walker checkpoint (its run's `config.json` beside it); its docstring lists every
option. Every step:

- **is anchored to the starting walker** (`common/anchor_ppo.py`; the anchor is the policy the fine-tune is held
  close to): PPO plus 5 × the mean squared difference between the
  actor's mean and that of the starting checkpoint, only on commands inside the range that walker was trained on (0.12
  m/s, 1.0 rad/s turning in place, 0.6 rad/s while walking) and not within 2 s of a kick. The walker keeps its gait where
  it already walked and learns the rest;
- trains on rough ground and stairs, narrow flights with 25-45 mm risers among them (`walker/terrain.py`), with foot
  clearance, stumble, progress and overspeed terms;
- kicks the robots: 0-0.6 m/s every 4-8 s, and 0.6-1.5 m/s every 8-15 s;
- puts the lidar on half the robots;
- holds the abductions' stand 12° further out than the calibrated stand (`--stand-offset-abd 12`): a wider stance is what
  makes the walker ride out shoves. The policy's meta carries the offset, and the runner applies it.

## Exporting and evaluating

```sh
.venv/bin/python -m mp4_rl.common.export runs/walker_step4/<run>/model_<k>.pt runs/exports/walker
.venv/bin/python -m mp4_rl.getup.runner check --policy runs/exports/walker

WALKER=runs/exports/walker sh skills/walker/eval/sku2.sh runs/eval_walker omni_s21 push_s21
.venv/bin/python skills/walker/eval/score.py runs/eval_walker
```

`skills/walker/eval/sku2.sh` runs the walker's evaluations on the SKU2's plant (all of them take a few hours on one GPU);
the names select some of them:

| Name | Evaluation |
|---|---|
| `omni_s21`, `omni_s22` | the omni command grid (walks in every direction, turns, walks while turning; 24 robots per command), seeds 21 and 22 |
| `omni_lidar_s21`, `omni_lidar_s22` | the same with the lidar on the hood |
| `push_s21`, `push_s22` | kicks while standing and while walking |
| `classic_s21`, `classic_s22` | the firmware's classic gait on the omni grid, the baseline |
| `wrough_s21`, `wrough_lidar_s21` | rough ground, without and with the lidar |
| `wstairs_s21`, `wstairs_lidar_s21` | the printed four-step stairs, without and with the lidar |
| `vid_side`, `vid_front`, `vid_stairs` | short videos (with `VIDEOS=1`) |

`score.py` prints the tables: the speed on each command against the classic gait, falls per kick speed, success on rough
ground and the stairs. The export takes the robot from the run (`--robot` exports for another one), and an evaluator
takes the robot, legs and joint limits from the policy's metadata.

On an RTX 5080 one evaluation takes a few minutes (an omni grid about 1-2 min); the kick grid and the terrain
evaluations run longer, and the whole script a few hours. A name runs one evaluation. `QUICK=1` runs each with an eighth
of its robots, as a quick check rather than a result (give it its own `OUT`: a finished evaluation's JSON is not run
again):

```sh
QUICK=1 WALKER=runs/exports/walker sh skills/walker/eval/sku2.sh runs/eval_walker_quick omni_s21 push_s21
```

**Checking a re-export.** `export --from-policy policies/sku2 OUT` gives the same network arrays and metadata (its
`source_policy_sha256` records the folder's `policy.npz`) and byte-identical copies of `config.json`, the calibration
and the joint limits. An export of the final checkpoint (`checkpoints/sku2/walker/model_10793.pt`) gives the same
network arrays and metadata except `source_policy_sha256`; the `policy.npz` bytes and `config.json` differ, because the
shipped folder is itself a `--from-policy` copy. For the SKU1 walker, `--from-policy policies/sku1` gives the same
arrays and metadata (plus `source_policy_sha256`) and byte-identical copies of the other files, and an export of
`checkpoints/sku1/walker/model_11792.pt` gives the shipped `policy.npz` byte for byte (its `config.json` then holds only
the checkpoint's options).
`tests/cpu/test_policies.py` checks all of them against the shipped folders:

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/cpu/test_policies.py -k export
```

**On the SKU1:** `skills/walker/eval/sku1.sh` runs the same evaluations on the SKU1's plant, by default for
`policies/sku1`; `WALKER` selects another walker exported for the SKU1, e.g. the SKU2 walker as it is:

```sh
sh skills/walker/eval/sku1.sh runs/eval_walker_sku1 omni_s21 push_s21
.venv/bin/python skills/walker/eval/score.py runs/eval_walker_sku1

.venv/bin/python -m mp4_rl.common.export --from-policy skills/walker/policies/sku2 runs/exports/walker_sku1 --robot sku1
WALKER=runs/exports/walker_sku1 sh skills/walker/eval/sku1.sh runs/eval_walker_sku2_on_sku1 omni_s21
```

**Simulation results** come from these evaluations: each JSON in `OUT` holds one evaluation's figures, and `score.py`
prints the tables. A result depends on the seed and varies by a few points between runs on a GPU.

## On the robot

On the SKU2 the walker runs on the Raspberry Pi with the get-up's runner (`mp4_rl.getup.runner`, which runs both
policies and switches between them): see the get-up's [SKU2 robot guide](../getup/docs/ROBOT_SKU2.md). On the SKU1 both
policies run on the ESP32 with the Mini Pupper v4 firmware: see the [SKU1 robot guide](../getup/docs/ROBOT_SKU1.md).
