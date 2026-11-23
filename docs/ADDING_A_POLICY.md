# Adding a policy

A new policy needs its folder under `mp4_rl/skills/`, a line in the registry, rows in the test tables that pin the
shipped files, and, if it is to run on the SKU2 through the runner, a policy kind the runner knows and a line in
`tools/deploy.sh`. The walker ([`mp4_rl/skills/walker/`](../mp4_rl/skills/walker/)) and the get-up
([`mp4_rl/skills/getup/`](../mp4_rl/skills/getup/)) are the worked examples: copy the one closer to yours.

## 1. The folder

`mp4_rl/skills/<name>/`, a Python package (`__init__.py`) holding everything about the policy:

| Part | What it holds | Walker | Get-up |
|---|---|---|---|
| `README.md` | the policy's page: animations, what it is, the policies table, how the pieces fit, on the robot, quick start, the recipe chain, results on the robot, evaluation and scoring, re-export check, what is where | [README.md](../mp4_rl/skills/walker/README.md) | [README.md](../mp4_rl/skills/getup/README.md) |
| `media/` | the animations the page shows (kept out of the wheel) | walking, stairs, rough ground per robot | get-ups and shoves per robot |
| `task.py` | the task on `mp4_rl.common.env` (the base environment on `mp4_sim`'s scene): commands or starts, rewards, terminations, curricula | `task.py`, with its training terrain in `terrain.py` | `task.py` |
| `train.py` | the trainer, `python -m mp4_rl.skills.<name>.train`, on PPO from `mp4_rl.common.algorithms` (or `common.anchor_ppo` to hold it near another policy) | anchored PPO | PPO with the smoothness loss |
| `eval/` | the evaluators (`python -m mp4_rl.skills.<name>.eval.<module>`), one script per robot (`sku2.sh`, `sku1.sh`) that writes one JSON per evaluation, and `score.py` that prints the tables | `omni.py`, `push.py`, `terrain.py` | `getup_eval.py`, `sequence.py` |
| `recipes/<robot>/` | shell scripts with every training setting, run from the repository root | `sku2/step1.sh` ... `step4.sh`, `sku1/finetune.sh` | `sku2/finetune.sh`, `sku1/finetune.sh` |
| `checkpoints/<robot>/` | where the recipes start and their last checkpoints, each with a minimal `config.json` (the options `--resume` and the exporter read) | `sku2/walker_start/`, `sku2/walker/`, `sku1/walker/` | `sku2/getup_start/`, `sku2/getup/`, `sku1/getup/` |
| `policies/<robot>/` | the exported policy: `python -m mp4_rl.common.export CHECKPOINT mp4_rl/skills/<name>/policies/<robot>` writes `policy.npz`, `config.json` and copies of the calibration and joint limits | `sku2/`, `sku1/` | `sku2/`, `sku1/` |
| `docs/` | anything else the policy needs, such as running it on the robot | | `ROBOT_SKU2.md`, `ROBOT_SKU1.md` |

Shared code goes in `mp4_rl/common/` (or `mp4_rl/deploy/` for the robot side), not in a policy's folder.
`tests/cpu/test_bank.py` expects every part above but `media/` and `docs/`, and `policies/`, `recipes/`, `checkpoints/`
and an `eval/<robot>.sh` for each robot in `ROBOTS` (`sku2`, `sku1`).

## 2. The registration line

A line in `REGISTRY` in [`mp4_rl/skills/__init__.py`](../mp4_rl/skills/__init__.py), the name and what the policy does:

```python
REGISTRY = {
    "walker": "walks in every direction, turns in place and rides out shoves",
    "getup": "gets the robot back on its feet from any fall pose and hands back to the walker",
    "<name>": "<what it does>",
}
```

The install check (`scripts/install/verify.py`) then checks each of its `policies/<robot>/` folders against `mp4_sim`'s
files, and `tests/cpu/test_bank.py` checks that the folder is registered and holds its parts.

## 3. The tests, the provenance and the index

- `tests/cpu/test_policies.py`: a row per policy and robot in `ARRAYS` (the network arrays' hash), `NPZ` (the policy
  file's sha256) and `FINAL` (its final checkpoint), and a row per checkpoint, keyed by the policy's name, in
  `CHECKPOINTS` (iteration, observation normaliser) and `SHA` (sha256). The re-export and checkpoint tests then cover
  it. Its recipe tests (`test_the_sku1_configs_hold_the_options_their_recipes_build` and the recipe command-line tests)
  name the walker's and the get-up's trainers: add a case for the new trainer if its recipes should be checked too.
- A row per shipped file in [PROVENANCE.md](../PROVENANCE.md).
- A row in the root [README.md](../README.md)'s table: a thumbnail from `media/`, what it does, its robots, a link to its
  folder.

## 4. On the robot (SKU2, through the runner)

The runner (`mp4_rl/deploy/runner/`) runs a walking policy and a get-up policy, chosen by the meta `kind` an export
records: `walk` or `getup` (`POLICY_KINDS` in `mp4_rl/common/policy.py`, [POLICY_CONTRACT.md](POLICY_CONTRACT.md)). A
policy of one of these kinds, with the inputs and outputs that document describes, runs with `--policy DIR` (`walk`) or
`--getup-policy DIR` (`getup`). A new kind
needs its own code in the runner (`runner.py`, `checks.py`) and its tests (`tests/cpu/test_runner_*.py`).
`tools/deploy.sh` copies the walker's and the get-up's SKU2 folders (its `for skill in walker getup` loop and its
`CHECK` line); add the new folder there to deploy it, and to `test_deployed_subset_needs_only_numpy_and_pyserial` in
`tests/cpu/test_tools.py`, which checks the deployed folder's exact contents.

## 5. Check it

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/cpu
.venv/bin/python scripts/install/verify.py cpu
```
