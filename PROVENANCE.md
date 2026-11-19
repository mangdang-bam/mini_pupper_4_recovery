# Provenance

What the shipped files are and where they came from.

| File | What it is | sha256 |
|---|---|---|
| `policies/walker/policy.npz` | The walker that ran on the robot: `checkpoints/walker/model_10793.pt` exported (`export_policy`), the abductions' stand 12° wider | `5f7f2d70864c377c87aef3ae7546ed9def5c7deac862e21b22644c6a083372dd` |
| `policies/getup/policy.npz` | The get-up that ran on the robot: `checkpoints/getup/model_7543.pt` exported | `6b537110f8924d66a648882b2ddc24b065a39450b718c9731f1ab57b91be5c6d` |
| `checkpoints/walker_start/model_6297.pt` | The walker's starting checkpoint: descends from the stair-and-rough-ground walker training of [mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim), trained further on rough ground and stairs | `33cc59eb3bf1c7384dea50263c719b522cae5bece5997fcf87acbf879f42c4ca` |
| `checkpoints/walker/model_10793.pt` | The walker's final checkpoint: `walker_round1.sh` to `walker_round4.sh` from the starting checkpoint (seeds 71, 81, 91, 101) | `290793f29f483421b0504a752cfdf4fb7b461ad37fbd0b48882f6ea9affd2712` |
| `checkpoints/getup_start/model_6544.pt` | The get-up's starting checkpoint: a get-up trained with the earlier servo model fit (`md01`) and a 478.66 g mass set, with a load on the hood and uneven ground | `690dd19d9e627e224d07bd581dae399c79b7e94afaf274d128bed8aa889f6c99` |
| `checkpoints/getup/model_7543.pt` | The get-up's final checkpoint: `getup_finetune.sh` from the starting checkpoint (seed 72) | `2ebb10d46af9561ee684a13362a5febf3afc734eba9cfa94abfc22e223f9ddc7` |

The two starting checkpoints are where the recipes begin; the recipes that made them are not in this repository.

Each checkpoint's `config.json` holds only the run options that `--resume` and `export_policy` read. Each policy's meta records
its checkpoint's sha256 (`checkpoint_sha256`) and `source_policy_sha256`, which `export_policy --from-policy` records: the
hash of the policy file the weights were copied from.

The robot model, the fitted servo parameters, the installer and the servo model code come from mini_pupper_4_sim; the
servo model builds on [BAM](https://github.com/Rhoban/bam). The masses were weighed and the joint limits measured by hand on one robot.
