# Adding a policy

A policy is one folder under `mp4_rl/skills/` and one line in the registry. The walker
([`mp4_rl/skills/walker/`](../mp4_rl/skills/walker/)) and the get-up ([`mp4_rl/skills/getup/`](../mp4_rl/skills/getup/))
are the worked examples: copy the one closer to yours.

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

Shared code goes in `mp4_rl/common/` (or `mp4_rl/deploy/` for the robot side), not in a policy's folder. A policy that
runs on the SKU2 through the runner (`mp4_rl/deploy/runner/`) has to be of a kind the runner knows: `walk` or `getup`
([POLICY_CONTRACT.md](POLICY_CONTRACT.md)).

## 2. The registration line

One line in `REGISTRY` in [`mp4_rl/skills/__init__.py`](../mp4_rl/skills/__init__.py), the name and what the policy does:

```python
REGISTRY = {
    "walker": "walks in every direction, turns in place and rides out shoves",
    "getup": "gets the robot back on its feet from any fall pose and hands back to the walker",
    "<name>": "<what it does>",
}
```

The install check (`scripts/install/verify.py`) then checks each of its `policies/<robot>/` folders against `mp4_sim`'s
files, and `tests/cpu/test_bank.py` checks that the folder is registered and holds its parts.

## 3. The index and the tests

- A row in the root [README.md](../README.md)'s table: a thumbnail from `media/`, what it does, its robots, a link to its
  folder.
- Its shipped files pinned in `tests/cpu/test_policies.py` (`ARRAYS`, `NPZ`, `FINAL`, `CHECKPOINTS`, `SHA`), so the
  re-export checks cover it, and a row per shipped file in [PROVENANCE.md](../PROVENANCE.md).

Check it:

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/cpu/test_bank.py tests/cpu/test_policies.py
.venv/bin/python scripts/install/verify.py cpu
```
