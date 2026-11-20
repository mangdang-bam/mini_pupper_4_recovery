# The get-up

A get-up policy for the Mini Pupper v4: puts the robot back on its feet from any fall pose, with or without the lidar on
the hood. On the robot it runs together with the [walker](../walker/README.md): the runner on the Raspberry Pi switches
to the get-up on a detected fall and back to the walker once the robot stands.

| Robot | Policy | Inputs | Status |
|---|---|---|---|
| SKU2 | [`policies/sku2`](policies/sku2) | 45 | ran on the robot |
| SKU1 | [not available yet](policies/sku1/README.md) | | recipe and evaluation in simulation |

The policy folder holds `policy.npz`, `config.json` and copies of the calibration and joint limits it was exported for;
the runner reads the calibration from there ([POLICY_CONTRACT.md](../../docs/POLICY_CONTRACT.md)).

## On the robot

[docs/ROBOT_SKU2.md](docs/ROBOT_SKU2.md): the firmware, deploying to the Pi (`tools/deploy.sh`), the checks before a run,
launching, stopping, the safety rules and what was measured. The runner (`src/mp4_rl/getup/runner/`) and the servo link
(`src/mp4_rl/getup/mp4link/`) need numpy and pyserial only.

## How it was made

Fine-tuned once on the SKU2's simulated plant (the simulated robot as training and evaluation see it) from `checkpoints/sku2/getup_start/model_6544.pt` (the plant as for the
walker: the SKU2 model with its measured joint limits and weighed masses, the MD-300 servo model on a 3S supply, from
[mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim)):

```sh
sh skills/getup/recipes/sku2/finetune.sh     # seed 72, 1000 iterations; output in runs/getup_finetune/
```

It takes `[SEED] [PARENT] [ITERS]`, and `ENVS`, `OUT` and `EXTRA` from the environment. `checkpoints/sku2/getup/model_7543.pt`
is its last checkpoint: `policies/sku2` was exported from it. On an RTX 5080 the fine-tune takes 2.5-2.7 h.

**A quick check of a recipe** (fewer robots and a few iterations, in its own output folder). The positional arguments
are `[SEED] [PARENT] [ITERS]`; an empty one keeps its default:

```sh
ENVS=64 OUT=runs/quick sh skills/getup/recipes/sku2/finetune.sh "" "" 5
```

A run saves `model_<k>.pt` where k counts on from the parent's iteration: every 50 iterations (k a multiple of 50) and
at the end, k = the parent's iteration + ITERS - 1 (`model_6548.pt` from `model_6544.pt` with 5 iterations). Such a
checkpoint exports and passes `runner check` like any other; it is not a trained policy.

### The SKU1

`recipes/sku1/finetune.sh` fine-tunes the same starting checkpoint on the SKU1's simulated plant with the SKU2
fine-tune's settings (the trainer's table under [The trainer](#the-trainer)), except:

| Recipe | Parent | Seed | Iterations |
|---|---|---|---|
| `sku1/finetune.sh` | `checkpoints/sku2/getup_start/model_6544.pt` | 71 | 1000 |

- the SKU1 (`--robot sku1`): its model, calibration, joint limits traced on the robot and weighed masses (413.4 g);
- its display as a collision box on the body (`--display-collision`);
- instead of the random load, the SKU1's lidar (35 mm high, 47.2 g with its mount, its axis drawn 20 mm behind to 10 mm
  in front of the base origin and 10 mm either side) on half the robots;
- the whole robot's centre of mass drawn 61-70 mm above the floor.

```sh
sh skills/getup/recipes/sku1/finetune.sh 71     # output in runs/getup_sku1/
```

### The trainer

`python -m mp4_rl.getup.train` trains the get-up; with `--resume` it fine-tunes a checkpoint (actor, critic, optimiser and
iteration come from it). The fine-tune keeps the starting checkpoint's network (512-256-128, with the normaliser) and its
settings:

| Setting | Value |
|---|---|
| Starts | crouched, partly fallen, on a side, nose down, on the back, random joints (5, 5, 25, 15, 40, 10 %); a fifth from a mid-fall |
| Episode | 8 s; ends once the robot is up |
| Band | 90° on abductions, 180° on hips and knees, about the stand, inside the measured joint limits |
| Phase clock | 4 s |
| Target filter | α 0.75 (see [POLICY_CONTRACT.md](../../docs/POLICY_CONTRACT.md#target-filter)) |
| Costs | action rate −0.1, kept whole while tilted; joint velocity −5e-4 |
| Terrain | half the columns flat, half rough ground (relief on a 2 cm grid, slopes, boxes) and the printed four- and eight-step stairs |
| Load on the hood | a cylinder drawn per robot (0-40 mm high, 0-70 g) |
| Exploration | the checkpoint's std (0.03), clamped to 0.06-0.3 when sampling; no entropy bonus |
| Learning rate | starts at 1e-5, adaptive, at most 5e-5; the critic warmed up for 30 iterations |

**The smoothness loss** (the trainer's `--caps-*` options, after CAPS, Conditioning for Action Policy Smoothness). Added
to the PPO update (`common/algorithms.py`), on the actor's mean, with weights 2 (temporal)
and 40 (spatial):

```
temporal  mean |mu(o_t+1) - mu(o_t)|^2      over consecutive steps of one robot, not across a reset
spatial   mean |mu(o) - mu(o + eps)|^2      eps ~ N(0, 0.05^2), in the normalised input
```

With the target filter it removes fast target reversals while the robot is in the air. Both have to be trained under: a
filter added to a finished policy breaks it.

## Exporting and evaluating

```sh
.venv/bin/python -m mp4_rl.common.export runs/getup_finetune/<run>/model_<k>.pt runs/exports/getup
.venv/bin/python -m mp4_rl.getup.runner check --policy runs/exports/getup

GETUP=runs/exports/getup sh skills/getup/eval/sku2.sh runs/eval_getup gflat_s21 glidar_s21 seq_s21
.venv/bin/python skills/getup/eval/score.py runs/eval_getup

# Videos: watch them, the numbers don't show everything
VIDEOS=1 GETUP=runs/exports/getup sh skills/getup/eval/sku2.sh runs/eval_getup vid_back
```

`export` takes the robot, the joint limits, the stance offset, the target filter and the servo model from the run's
options; `--help` lists the overrides. `skills/getup/eval/sku2.sh` runs the get-up's evaluations (all of them take a few
hours on one GPU); the names select some of them, and `score.py` prints the tables:

| Name | Evaluation |
|---|---|
| `gflat_s21` ... `gflat_s24` | 512 robots from each start (left, right, back, nose down, mid-fall, random) on flat ground, seeds 21-24 |
| `glidar_s21` ... `glidar_s24` | the same with the lidar on the hood |
| `seq_s21` ... `seq_s24` | the sequence with the walker: walk, kicks of 0.5-1.5 m/s every 2-4 s, fall, get up, walk (256 robots, 60 s) |
| `gterr_s21`, `gterr_s23` | every start on rough ground (reliefs, slopes, boxes; 1024 robots) |
| `gstairs_s21`, `gstairs_s23` | every start on the printed four- and eight-step stairs (1024 robots) |
| `vid_back` | a short video of a get-up from the back (with `VIDEOS=1`) |

On an RTX 5080 one evaluation takes a few minutes (a get-up from every start on flat ground about 2-7 min); the sequence
and the terrain evaluations run longer, and the whole script a few hours. A name runs one evaluation. `QUICK=1` runs
each with an eighth of its robots, as a quick check rather than a result (give it its own `OUT`: a finished evaluation's
JSON is not run again):

```sh
QUICK=1 GETUP=runs/exports/getup sh skills/getup/eval/sku2.sh runs/eval_getup_quick gflat_s21
```

**Checking a re-export.** `export --from-policy policies/sku2 OUT` gives the same network arrays and metadata (its
`source_policy_sha256` records the folder's `policy.npz`) and byte-identical copies of `config.json`, the calibration
and the joint limits. An export of the final checkpoint (`checkpoints/sku2/getup/model_7543.pt`) gives the same network
arrays and metadata except `source_policy_sha256`; the `policy.npz` bytes and `config.json` differ, because the shipped
folder is itself a `--from-policy` copy. `tests/cpu/test_policies.py` checks both against the shipped folders:

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/cpu/test_policies.py -k export
```

**On the SKU1:** no get-up has been fine-tuned for the SKU1 yet. The SKU1 recipe starts from the get-up's starting
checkpoint, so that checkpoint exported for the SKU1 (the network as it is, the decoder's clamps from the SKU1's
calibration and joint limits) is the baseline a SKU1 fine-tune is compared with. `skills/getup/eval/sku1.sh` evaluates
it; it needs `GETUP`, and `WALKER` for the sequence (a walker exported for the SKU1, see [the walker](../walker/README.md)):

```sh
.venv/bin/python -m mp4_rl.common.export skills/getup/checkpoints/sku2/getup_start/model_6544.pt \
    runs/exports/getup_sku1 --robot sku1 --limits
GETUP=runs/exports/getup_sku1 sh skills/getup/eval/sku1.sh runs/eval_getup_sku1 gflat_s21 glidar_s21
.venv/bin/python skills/getup/eval/score.py runs/eval_getup_sku1
```

`--limits` puts the measured joint limits in the decoder's clamps. The starting checkpoints were trained without them
(their `config.json` has no `limits` option), so their export needs it; an export of a fine-tuned run (trained with
the limits) or of a policy folder (`--from-policy`) takes it from the run or the folder.

An exported policy folder converts the same way: `python -m mp4_rl.common.export --from-policy DIR OUT --robot sku1`.
The recipe trains with the display as a collision box; to evaluate with it, add `--display-collision` to the
evaluator's command line (a policy exported from a run with it has it in its metadata).

**Simulation results** are not published in this repository: run the evaluations above to produce the tables. A result
depends on the seed and varies by a few points between runs on a GPU.
