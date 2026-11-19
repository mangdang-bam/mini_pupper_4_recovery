# Vendored BAM (better-actuator-models)

Apache-2.0, see `LICENSE`. Copyright 2025 Marc Duclusaud & Grégoire Passault;
`mangdang/actuator.py` additionally 2026 Mangdang.

Source: the `bam` package of [BAM](https://github.com/Rhoban/bam), with
MangDang's servo model added (`mangdang/`: the MD-300-150 servo module, which its firmware calls MD01), and
`MD300Actuator` (another model of the same module) from Rhoban/bam pull request #26 at 2d1688a.

Vendored: the model and MD01 actuator (`model.py`, `actuator.py`, `parameter.py`, `testbench.py`, `mangdang/`) and the mjlab / MuJoCo
Warp binding (`mjlab.py`, with `simulate.py` and `testbench_mujoco.py`).

Changes from upstream:

- `bam.*` imports are package-relative; docstrings and messages point at `mp4_recovery.actuators.bam`.
- `actuators.py` registers only `md01v`, `md01vp` and `mangdang_md300_id_{1,2,3}`.
- `params/` is not vendored. Fitted parameters live in `mp4_recovery/data/` and are passed by path; the MD-300-150
  m6 sets are copied byte for byte as `data/md300_id_{1,2,3}_m6.json`.

Not vendored: the other servo families, servo fitting and bench tools (`fit.py`, `mae.py`, `plot.py`, `mangdang/tools`, recorders),
and their hardware dependencies.

Known difference: `mjlab.py`'s `_compute_friction_budget` adds the m6 quadratic load-friction term without the opposite-sign
condition that `Model.compute_frictions` in `model.py` applies. The servo (`mp4_recovery.actuators.bam_servo`) inherits this.
