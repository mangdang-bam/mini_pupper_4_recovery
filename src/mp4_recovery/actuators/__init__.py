"""Fitted servo (actuator) models.

- ``bam``          vendored BAM: models of MangDang's MD-300-150 servo module (called MD01 in its firmware) and the
                   mjlab binding
- ``bam_servo``    the GPU servo: BAM on mjlab joints and knee tendons
- ``servo_model``  the actuator set for the environments: ``md300`` (the default, the fit the shipped policies use) or
                   ``md01`` (an earlier fit), with the fitted parameters in ``mp4_recovery/data/md300_id_*_m6.json``
                   and ``mp4_recovery/data/md01_*.json``
- ``choice``       the model's names, gains and flags, without the simulator
"""
