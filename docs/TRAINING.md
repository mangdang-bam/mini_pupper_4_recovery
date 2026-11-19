# Training

Both policies are fine-tuned on the robot's simulated plant from the starting checkpoints in `checkpoints/`: the walker in four
rounds, the get-up once. The plant is the SKU2 model with its measured joint limits and weighed masses (630.5 g,
`src/mp4_recovery/data/sku2_masses.json`), and the MD-300 servo model on a 3S supply (10.5-12.6 V) with the winding resistance drawn from
0.95-1.15 × its fitted value. Training and evaluation need the `gpu` install ([INSTALL.md](INSTALL.md)). Run the recipes from
the repository root; each script's header lists every setting.

```sh
sh scripts/recipes/walker_round1.sh      # from checkpoints/walker_start/model_6297.pt
sh scripts/recipes/walker_round2.sh      # from the last checkpoint of the newest round-1 run, and so on to walker_round4.sh
sh scripts/recipes/getup_finetune.sh     # from checkpoints/getup_start/model_6544.pt
```

Each script takes `[SEED] [PARENT] [ITERS]`, and `ENVS`, `OUT` and `EXTRA` from the environment; its output goes to
`runs/<recipe name>/`.

## The settings used

| Recipe | Parent | Seed | Iterations | What it changes |
|---|---|---|---|---|
| `walker_round1.sh` | `checkpoints/walker_start/model_6297.pt` | 71 | 1000 | Widens the commands over the first 300 iterations, from 0.12 m/s and 1.0 rad/s to 0.20 m/s and 2.0 rad/s turning in place; kicks up to 1.5 m/s from iteration 300 |
| `walker_round2.sh` | round 1's last checkpoint | 81 | 1500 | The full command range from the start, 40 % of the walks and turns drawn from the fast end, and a reward for holding the heading |
| `walker_round3.sh` | round 2's last checkpoint | 91 | 1000 | Turning in place weighted up to 40 % of the commands, fast turns drawn from 1.0-2.0 rad/s |
| `walker_round4.sh` | round 3's last checkpoint | 101 | 1000 | Pays for partial turns, so fast turns in either direction keep turning; a stronger heading hold |
| `getup_finetune.sh` | `checkpoints/getup_start/model_6544.pt` | 72 | 1000 | Ground with relief, slopes and boxes, the printed four- and eight-step stairs, a random load on the hood (0-40 mm high, 0-70 g), a fifth of the starts mid-fall |

`checkpoints/walker/model_10793.pt` and `checkpoints/getup/model_7543.pt` are the last checkpoints of round 4 and of the get-up
fine-tune: `policies/walker` and `policies/getup` were exported from them.

On an RTX 5080 (16 GB) running three runs at a time, 1000 walker iterations take about 2-3 h and the get-up fine-tune
2.5-2.7 h. These are the settings used for the shipped policies, not a promise of the same numbers: a run's result depends on its
seed and the machine.

## The walker

`python -m mp4_recovery.gpu.train_walker` resumes a walker checkpoint (its run's `config.json` beside it); its docstring lists
every option. Every round:

- **is anchored to the starting walker** (`gpu/anchor_ppo.py`): PPO plus 5 × the mean squared difference between the actor's
  mean and that of `checkpoints/walker_start/model_6297.pt`, only on commands inside the range that walker was trained on (0.12
  m/s, 1.0 rad/s turning in place, 0.6 rad/s while walking) and not within 2 s of a kick. The walker keeps its gait where it
  already walked and learns the rest;
- trains on rough ground and stairs, narrow flights with 25-45 mm risers among them, with foot clearance, stumble, progress and
  overspeed terms;
- kicks the robots: 0-0.6 m/s every 4-8 s, and 0.6-1.5 m/s every 8-15 s;
- puts the lidar on half the robots;
- holds the abductions' stand 12° further out than the calibrated stand (`--stand-offset-abd 12`): a wider stance is what makes
  the walker ride out shoves. The policy's meta carries the offset, and the runner applies it.

## The get-up

`python -m mp4_recovery.gpu.train` trains the get-up; with `--resume` it fine-tunes a checkpoint (actor, critic, optimiser and
iteration come from it). The fine-tune keeps the starting checkpoint's network (512-256-128, with the normaliser) and its settings:

| Setting | Value |
|---|---|
| Starts | crouched, partly fallen, on a side, nose down, on the back, random joints (5, 5, 25, 15, 40, 10 %); a fifth from a mid-fall |
| Episode | 8 s; ends once the robot is up |
| Band | 90° on abductions, 180° on hips and knees, about the stand, inside the measured joint limits |
| Phase clock | 4 s |
| Target filter | α 0.75 (see [POLICY_CONTRACT.md](POLICY_CONTRACT.md#target-filter)) |
| Costs | action rate −0.1, kept whole while tilted; joint velocity −5e-4 |
| Exploration | the checkpoint's std (0.03), clamped to 0.06-0.3 when sampling; no entropy bonus |
| Learning rate | starts at 1e-5, adaptive, at most 5e-5; the critic warmed up for 30 iterations |

**The smoothness loss.** Added to the PPO update, on the actor's mean, with weights 2 (temporal) and 40 (spatial):

```
temporal  mean |mu(o_t+1) - mu(o_t)|^2      over consecutive steps of one robot, not across a reset
spatial   mean |mu(o) - mu(o + eps)|^2      eps ~ N(0, 0.05^2), in the normalised input
```

With the target filter it removes fast target reversals while the robot is in the air. Both have to be trained under: a filter
added to a finished policy breaks it.

## Exporting and checking a result

```sh
# Export a checkpoint (its run's config.json beside it) as a policy
python -m mp4_recovery.export_policy runs/getup_finetune/<run>/model_<k>.pt runs/exports/getup
python -m mp4_recovery.deploy.runner check --policy runs/exports/getup

# Evaluate it as the shipped policies are evaluated, then print the tables
GETUP=runs/exports/getup sh scripts/eval/sku2.sh runs/eval_getup gflat_s21 glidar_s21 seq_s21
python3 scripts/eval/score.py runs/eval_getup

# Videos: watch them, the numbers don't show everything
VIDEOS=1 GETUP=runs/exports/getup sh scripts/eval/sku2.sh runs/eval_getup vid_back
```

`export_policy` takes the joint limits, the stance offset, the target filter and the servo model from the run's options;
`--help` lists the overrides. `WALKER=DIR` evaluates another walker the same way: `omni_s21`, `push_s21`, `wrough_s21` and
`wstairs_s21` are its evaluations, and `scripts/eval/sku2.sh`'s header lists them all. `score.py` compares the walker with the
classic gait (`classic_s21`, `classic_s22`) on the same commands.
