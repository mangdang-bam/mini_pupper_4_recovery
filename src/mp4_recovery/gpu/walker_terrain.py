"""The walker's training terrain (the ``narrow`` mix the walker was trained on), on this package's surfaces.

- ``JitterPyramidStairsCfg``: square stairs around the spawn, every riser and tread drawn (pits climbing, pyramids
  descending); each tile's outermost riser is recorded in ``extents`` for ``walker_task.stair_exit``.
- ``NarrowFlightCfg``: narrow straight flights (4-5 risers, 17-30 cm wide, side drops, a landing), drawn per tile; each
  tile's flight is recorded in ``flights`` for ``walker_task.stair_exit`` and ``side_exit``. Built with ``StairGridCfg``
  (this package's measured-staircase builder: the drawn flight is its ``layout``).
- ``stairs7_sub_terrains`` and ``narrow_sub_terrains``: the walker's terrain mixes; its rough mix is this package's
  ``rough_sub_terrains`` (the same families, proportions and surfaces).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from mp4_recovery.gpu.terrain import STAIR_LAYOUTS, TILE_SIZE, StairGridCfg, rough_sub_terrains


@dataclass(kw_only=True)
class JitterPyramidStairsCfg(SubTerrainCfg):
    """Square stairs around the spawn, every riser and tread drawn. "climb": a pit, the spawn on its floor, rings rising
    outward to the border; "descend": a pyramid, the spawn on its top, rings falling outward to the floor. Per tile: the
    nominal riser ``riser_range[0] + difficulty * (riser_range[1] - riser_range[0])``, a nominal tread U(``tread_range``);
    per step the riser ``nominal * U(1 - jitter, 1 + jitter)`` and the tread ``nominal * U(0.8, 1.2)`` clamped to
    ``tread_range``; rings are added while they fit between the platform and the border. Solid boxes from z = 0."""
    riser_range: tuple[float, float] = (0.005, 0.060)
    tread_range: tuple[float, float] = (0.05, 0.20)
    jitter: float = 0.3
    direction: str = "climb"
    platform_width: float = 0.40
    border_width: float = 0.10

    def steps(self, difficulty: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(rises, treads) from the platform outward, in m."""
        h = self.riser_range[0] + difficulty * (self.riser_range[1] - self.riser_range[0])
        t_nom = rng.uniform(*self.tread_range)
        room = (self.size[0] - 2 * self.border_width - self.platform_width) / 2
        rises, treads = [], []
        while True:
            t = float(np.clip(t_nom * rng.uniform(0.8, 1.2), *self.tread_range))
            if sum(treads) + t > room + 1e-9:
                break
            treads.append(t)
            rises.append(h * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter))
        rises.append(h * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter))     # the border's riser
        return np.array(rises), np.array(treads)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        rises, treads = self.steps(difficulty, rng)
        self.last = (rises, treads)
        # the outermost riser's half-size per tile in generation order (one column's rows in order: the last num_rows
        # entries are rows 0..num_rows-1): where the stairs end (walker_task.stair_exit)
        self.__dict__.setdefault("extents", []).append(float(self.platform_width / 2 + treads.sum()))
        sx = self.size[0]
        c = sx / 2
        half = self.platform_width / 2 + np.concatenate([[0.0], np.cumsum(treads)])   # each riser's half-size
        body, geoms = spec.body("terrain"), []
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[c, c, 0.025], pos=[c, c, -0.025],
                          rgba=[0.5, 0.5, 0.5, 1.0])
        geoms.append(TerrainGeometry(geom=g))
        if self.direction == "descend":      # nested solid squares, the inner the taller
            tops = np.cumsum(rises[::-1])[::-1]
            for a, top in zip(half, tops):
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[a, a, top / 2], pos=[c, c, top / 2],
                                  rgba=[0.55, 0.40, 0.30, 1.0])
                geoms.append(TerrainGeometry(geom=g))
            origin = np.array([c, c, float(tops[0])])
        elif self.direction == "climb":      # nested square frames from each riser to the tile's edge, the outer taller
            tops = np.cumsum(rises)
            for a, top in zip(half, tops):
                w = c - a
                for pos, size in (([c, c + a + w / 2], [c, w / 2]), ([c, c - a - w / 2], [c, w / 2]),
                                  ([c + a + w / 2, c], [w / 2, a]), ([c - a - w / 2, c], [w / 2, a])):
                    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size + [top / 2], pos=pos + [top / 2],
                                      rgba=[0.55, 0.40, 0.30, 1.0])
                    geoms.append(TerrainGeometry(geom=g))
            origin = np.array([c, c, 0.0])
        else:
            raise ValueError(f"direction {self.direction!r}")
        return TerrainOutput(origin=origin, geometries=geoms)


@dataclass(kw_only=True)
class NarrowFlightCfg(StairGridCfg):
    """Narrow straight flights: per tile 4-5 risers of ``riser_range[0] + difficulty * (riser_range[1] - riser_range[0])``
    each times U(1 -+ ``jitter``), a nominal tread U(``tread_range``) times U(0.9, 1.1) per step (clamped), a width
    U(``width_range``) and a landing U(``landing_range``). The draws come in a fixed order; the flight is then built by ``StairGridCfg`` as its ``layout``. Each tile's flight, relative to its spawn, is
    appended to ``flights`` in generation order."""
    riser_range: tuple[float, float] = (0.010, 0.035)
    tread_range: tuple[float, float] = (0.045, 0.080)
    width_range: tuple[float, float] = (0.17, 0.30)
    landing_range: tuple[float, float] = (0.20, 0.40)
    steps_range: tuple[int, int] = (4, 5)
    jitter: float = 0.15
    tag: str = "narrow"

    def draw(self, difficulty: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, float]:
        """(rises (n,), treads between risers (n - 1,), width, landing) in m."""
        n = int(rng.integers(self.steps_range[0], self.steps_range[1] + 1))
        h = self.riser_range[0] + difficulty * (self.riser_range[1] - self.riser_range[0])
        rises = h * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter, n)
        t_nom = rng.uniform(*self.tread_range)
        treads = np.clip(t_nom * rng.uniform(0.9, 1.1, n - 1), *self.tread_range)
        width = float(rng.uniform(*self.width_range))
        landing = float(rng.uniform(*self.landing_range))
        return rises, treads, width, landing

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        rises, treads, self.width, self.landing = self.draw(difficulty, rng)
        self.layout = tuple((float(r), float(t)) for r, t in zip(rises, list(treads) + [0.0]))
        out = super().function(difficulty, spec, rng)
        L = STAIR_LAYOUTS[(self.tag, 0)]
        self.__dict__.setdefault("flights", []).append(
            {"x_first": float(L["risers_x"][0]), "x_last": float(L["risers_x"][-1]), "width": self.width,
             "landing": self.landing, "rises": L["rises"], "treads": L["treads"]})
        return out


NARROW_STAIR_FAMILIES = ("jstairs_climb", "jstairs_descend", "narrow")     # the pits are the jstairs_climb entry


def stairs7_sub_terrains(size: float = TILE_SIZE, stairs: float = 0.50, climb: float = 0.30) -> dict[str, SubTerrainCfg]:
    """The rough families without their box stairs, rescaled to ``1 - stairs``; jittered stairs (risers 5 -> 60 mm with
    difficulty, +-30 % per step, treads 50-200 mm) climbing ``climb`` and descending ``stairs - climb``."""
    subs = rough_sub_terrains(size)
    for k in ("stairs_up", "stairs_down"):
        del subs[k]
    rest = sum(c.proportion for c in subs.values())
    for c in subs.values():
        c.proportion *= (1.0 - stairs) / rest
    sz = (size, size)
    subs["jstairs_climb"] = JitterPyramidStairsCfg(proportion=climb, size=sz, direction="climb")
    subs["jstairs_descend"] = JitterPyramidStairsCfg(proportion=stairs - climb, size=sz, direction="descend")
    return subs


def narrow_sub_terrains(size: float = TILE_SIZE, narrow: float = 0.20, pits: float = 0.10,
                        riser_range: tuple[float, float] | None = None,
                        boxes_share: float | None = None) -> dict[str, SubTerrainCfg]:
    """``stairs7`` with its jittered climbs (30 %) split into narrow straight flights (``narrow``) and the square pits
    (``pits``); the descents and the rough families unchanged. ``riser_range`` (m, nominal at difficulty 0 and 1)
    replaces the narrow flights' default (0.010, 0.035); ``boxes_share``: the box obstacles' share of the total, the
    difference taken from the stair families in proportion."""
    subs = stairs7_sub_terrains(size)
    subs["jstairs_climb"].proportion = pits
    subs["narrow"] = NarrowFlightCfg(proportion=narrow, size=(size, size),
                                     **({"riser_range": tuple(riser_range)} if riser_range else {}))
    if boxes_share is not None:
        stairs = sum(subs[k].proportion for k in NARROW_STAIR_FAMILIES)
        extra = float(boxes_share) - subs["boxes"].proportion
        if not (0.0 <= float(boxes_share) and extra < stairs):
            raise ValueError(f"boxes_share {boxes_share}: from 0 to below the boxes' and stairs' "
                             f"{subs['boxes'].proportion + stairs:.4f}")
        for k in NARROW_STAIR_FAMILIES:
            subs[k].proportion *= (stairs - extra) / stairs
        subs["boxes"].proportion = float(boxes_share)
    return subs
