# Provenance

What the shipped files are and where they came from.

| File | What it is | sha256 |
|---|---|---|
| `skills/walker/policies/sku2/policy.npz` | The walker that ran on the robot: `skills/walker/checkpoints/sku2/walker/model_10793.pt` exported (`mp4_rl.common.export`), the abductions' stand 12° wider | `5f7f2d70864c377c87aef3ae7546ed9def5c7deac862e21b22644c6a083372dd` |
| `skills/getup/policies/sku2/policy.npz` | The get-up that ran on the robot: `skills/getup/checkpoints/sku2/getup/model_7543.pt` exported | `6b537110f8924d66a648882b2ddc24b065a39450b718c9731f1ab57b91be5c6d` |
| `skills/walker/checkpoints/sku2/walker_start/model_6297.pt` | The walker's starting checkpoint: it descends from an earlier stair-climbing walker, trained on an older model of the robot in [mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim), then trained further on rough ground and stairs | `33cc59eb3bf1c7384dea50263c719b522cae5bece5997fcf87acbf879f42c4ca` |
| `skills/walker/checkpoints/sku2/walker/model_10793.pt` | The walker's final checkpoint: `recipes/sku2/step1.sh` to `step4.sh` from the starting checkpoint (seeds 71, 81, 91, 101) | `290793f29f483421b0504a752cfdf4fb7b461ad37fbd0b48882f6ea9affd2712` |
| `skills/getup/checkpoints/sku2/getup_start/model_6544.pt` | The get-up's starting checkpoint: a get-up trained with the earlier servo model fit (`md01`) and an earlier mass estimate of the robot, with a load on the hood and uneven ground | `690dd19d9e627e224d07bd581dae399c79b7e94afaf274d128bed8aa889f6c99` |
| `skills/getup/checkpoints/sku2/getup/model_7543.pt` | The get-up's final checkpoint: `recipes/sku2/finetune.sh` from the starting checkpoint (seed 72) | `2ebb10d46af9561ee684a13362a5febf3afc734eba9cfa94abfc22e223f9ddc7` |
| `skills/*/policies/sku2/sku2_contract.json`, `sku2_joint_limits.json` | Byte copies of the SKU2's calibration and joint limits from mini_pupper_4_sim 0.2.0 (`robot/assets/sku_2/measured/`), the files each policy was exported for | `86f1d330a9f30d7e…`, `9cad8e76f4b8bf50…` |

The two starting checkpoints are given as they are: they are where the recipes begin, and their own training can't be
reproduced from this repository. No SKU1
policy or checkpoint is shipped: the SKU1 recipes (`skills/*/recipes/sku1/`) start from the SKU2 checkpoints above.

Each checkpoint's `config.json` holds only the run options that `--resume` and `mp4_rl.common.export` read. Each policy's
meta records its checkpoint's sha256 (`checkpoint_sha256`), the calibration's (`contract_sha256`), the joint limits'
(`bounds.info.limits_sha256`), the robot model's (`legs.model_sha256`), and `source_policy_sha256`, which
`mp4_rl.common.export --from-policy` records: the hash of the policy file the weights were copied from. Each policy's
`config.json` names the robot model, masses, joint limits and servo parameters it was trained on by their names in
mini_pupper_4_sim (`mp4_sim:...`) and their sha256.

## Data

| File | What it is |
|---|---|
| `src/mp4_rl/getup/data/sku2_proven_reach.json` | How far each hip has moved towards the body's middle on the robot, per abduction bin (the 99.5th percentile, from at least 20 samples): summarised from 28 runs on one SKU2 (809.5 s); the logs are not published. `mp4_rl.getup.eval.reach` compares a simulated get-up with it |
| `tests/cpu/data/feedback_glitches.npz` | Two excerpts of a SKU2 run log: a limp leg moved by hand with ten frames of impossible position readings, and a still stretch |
| `tests/cpu/data/sku2_imu_captures.npz` | Accelerometer captures from the SKU2 while trotting |
| `tests/cpu/data/firmware_trot_ticks.npz` | The firmware's trot (its gait and inverse kinematics, built on the host) tabulated at several speeds for one calibration |
| `tests/cpu/data/sku2_contract_test.json` | A calibration file with other offsets than the shipped one |
| `tests/data/compare_run/` | Small run and stream logs in the robot's log formats, for the tests of `tools/compare_run.py` |
| `tests/data/phase_restart_cross.json` | Scripted get-up cases played through the runner on a simulated board: the reference the GPU sequence test compares with |

The robot model, its measured data, the fitted servo parameters and the servo model code come from mini_pupper_4_sim;
the servo model builds on [BAM](https://github.com/Rhoban/bam). The masses were weighed and the joint limits measured by
hand on one robot.
