"""Multi-camera fusion: the grids of several cameras of one floor into one grid (E2a).

Every camera's grid estimates the same floor: where two cameras cover a cell,
both saw (part of) the same people, so adding them counts those people twice.
The fused value of a cell is a convex combination of the covering cameras'
values: weights are zero outside a camera's coverage and sum to one over the
cameras that cover the cell, so a person seen by every camera is counted once.

Weightings:
- sum: no weighting, every covering camera added (double counts; diagnostic);
- mean: every covering camera alike;
- visibility: a camera's weight grows with how much of the cell's traffic it
  sees. All covering cameras watch the same people there, so the ratio of a
  camera's dwell to the largest camera's dwell in that cell is its relative
  visibility (occlusion, distance, missed detections), without any label.
  Weight = ratio ** power: power 0 is the mean, power inf the camera that sees
  most. With zone_m, the ratio is measured on zone_m blocks of cells (less
  noise, coarser). With covered_cells, a camera's zone value is its dwell per
  zone cell it covers (the zone sum over the number of covered cells), so a
  camera that covers part of a zone is not penalised for the cells it cannot
  see (improvements plan, phase 4). covered_alpha blends the two: the zone
  value is divided by zone_cells ** (1 - alpha) * covered_cells ** alpha, so 0
  is the plain zone sum and 1 is covered_cells (improvements plan, phase 6, H1).

The weights come from whole-run (or whole-slot) grids and are applied
linearly, so fusing time blocks and adding them equals fusing the sum.
"""

import math
from dataclasses import dataclass

import numpy as np

from space_usage_heatmaps.occupancy.domain.grid import Grid

WEIGHTINGS = ("sum", "mean", "visibility")


@dataclass(frozen=True)
class CameraGrid:
    camera_id: str
    grid: Grid  # on the fused grid's spec
    coverage: np.ndarray  # (ny, nx) bool: cells whose centre projects inside its image


@dataclass(frozen=True)
class Fusion:
    weighting: str = "visibility"
    power: float = 1.0  # visibility: exponent of the relative visibility; math.inf allowed
    zone_m: float | None = None  # visibility: measured on blocks of zone_m metres
    covered_cells: bool = False  # visibility on zones: per covered cell of the zone
    covered_alpha: float = 0.0  # visibility on zones: partial covered_cells, 0 (off) to 1

    def __post_init__(self):
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"fusion weighting must be one of {WEIGHTINGS}: {self.weighting!r}")
        if not self.power >= 0:
            raise ValueError(f"fusion power must be 0 or more, not {self.power}")
        if self.zone_m is not None and not self.zone_m > 0:
            raise ValueError(f"fusion zone_m must be positive, not {self.zone_m}")
        if self.covered_cells and (self.weighting != "visibility" or self.zone_m is None):
            raise ValueError("fusion covered_cells needs the visibility weighting and zone_m")
        if not 0 <= self.covered_alpha <= 1:
            raise ValueError(f"fusion covered_alpha must be in [0, 1], not {self.covered_alpha}")
        if self.covered_alpha > 0 and (self.weighting != "visibility" or self.zone_m is None):
            raise ValueError("fusion covered_alpha needs the visibility weighting and zone_m")
        if self.covered_alpha > 0 and self.covered_cells:
            raise ValueError("fusion covered_alpha replaces covered_cells: set one of them")


DEFAULT = Fusion()


def joint_coverage(grids):
    """Cells at least one camera covers."""
    return np.logical_or.reduce([np.asarray(g.coverage, bool) for g in grids])


def overlap(grids):
    """Cells two or more cameras cover: where double counting can happen."""
    return np.sum([np.asarray(g.coverage, bool) for g in grids], axis=0) >= 2


def fusion_weights(grids, fusion=DEFAULT):
    """(cameras, ny, nx) weights; on the joint coverage they sum to one (sum:
    to the number of covering cameras), elsewhere they are zero."""
    if not grids:
        raise ValueError("no camera to fuse")
    spec = grids[0].grid.spec
    if any(g.grid.spec != spec for g in grids):
        raise ValueError("every camera grid must be on the same floor grid")
    cov = np.stack([np.asarray(g.coverage, bool) for g in grids])
    if fusion.weighting == "sum":
        return cov.astype(float)
    if fusion.weighting == "mean" or fusion.power == 0:
        w = cov.astype(float)
    else:
        seen = np.stack([np.where(c, g.grid.dwell, 0.0) for g, c in zip(grids, cov, strict=True)])
        if fusion.zone_m is not None:
            k = round(fusion.zone_m / spec.cell_m)
            seen = _zones(seen, k)
            if fusion.covered_cells:
                n = _zones(cov.astype(float), k)
                seen = np.divide(seen, n, out=np.zeros_like(seen), where=n > 0)
            elif fusion.covered_alpha > 0:  # 0: the plain zone sum, untouched
                a = fusion.covered_alpha
                n = _zones(cov.astype(float), k)
                cells = _zones(np.ones((1, *cov.shape[1:])), k)
                d = cells ** (1 - a) * n**a
                seen = np.divide(seen, d, out=np.zeros_like(seen), where=n > 0)
        top = seen.max(axis=0)
        ratio = np.divide(seen, top, out=np.zeros_like(seen), where=top > 0)
        if math.isinf(fusion.power):
            w = (cov & (ratio == 1)).astype(float)
        else:
            w = np.where(cov, ratio**fusion.power, 0.0)
        w = np.where(top > 0, w, cov)  # nobody seen by any camera: every covering one alike
    total = w.sum(axis=0)
    return np.divide(w, total, out=np.zeros_like(w), where=total > 0)


def _zones(a, k):
    """Each cell takes the sum of its k x k block (cameras, ny, nx).

    Known bias (phase 9 review, kept as chosen on tune and confirmed on sealed):
    a camera's block sum holds only the cells it covers (the others are zero in
    `seen`), so a camera that covers part of a block gets a lower ratio there
    than the camera covering all of it, even on the cells it sees well. This
    lowers the weights of cameras at the edges of their coverage. Fusion's
    covered_cells divides by the number of covered cells in the block instead
    (improvements plan, phase 4)."""
    # The default stays the sum; the mean over covered cells is opt-in
    # (covered_cells), a separate variant scored on tune.
    n, ny, nx = a.shape
    py, px = -(-ny // k) * k, -(-nx // k) * k
    padded = np.pad(a, ((0, 0), (0, py - ny), (0, px - nx)))
    sums = padded.reshape(n, py // k, k, px // k, k).sum(axis=(2, 4))
    return np.repeat(np.repeat(sums, k, axis=1), k, axis=2)[:, :ny, :nx]


def fuse_layers(weights, layers):
    """Sum over cameras of weight times layer; each layer (..., ny, nx), e.g.
    (blocks, ny, nx) time blocks of one camera."""
    return sum(w * np.asarray(layer, float) for w, layer in zip(weights, layers, strict=True))


def fuse_grids(grids, fusion=DEFAULT):
    """One Grid of the floor from the cameras' grids on the same spec (dropped:
    the cameras' samples outside the grid, added)."""
    w = fusion_weights(grids, fusion)
    return Grid(
        grids[0].grid.spec,
        fuse_layers(w, [g.grid.dwell for g in grids]),
        fuse_layers(w, [g.grid.passage for g in grids]),
        sum(g.grid.dropped for g in grids),
    )
