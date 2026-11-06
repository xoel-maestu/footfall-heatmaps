"""Floor grid with the two occupancy layers: dwell and passage."""

import math
from dataclasses import dataclass

import numpy as np

ZONE_M = 2.0  # largest block the metrics aggregate to; extents snap to it


@dataclass(frozen=True)
class GridSpec:
    """Cells of cell_m metres; row i spans y in [origin_y + i*cell_m, ...)."""

    origin_x: float
    origin_y: float
    cell_m: float
    nx: int
    ny: int

    def __post_init__(self):
        if self.cell_m <= 0 or not math.isclose(1 / self.cell_m, round(1 / self.cell_m)):
            raise ValueError(f"cell_m must divide 1 m exactly, got {self.cell_m}")

    @classmethod
    def covering(cls, xmin, ymin, xmax, ymax, cell_m):
        """Smallest grid of whole 2 m blocks that contains the box."""
        x0, y0 = math.floor(xmin / ZONE_M) * ZONE_M, math.floor(ymin / ZONE_M) * ZONE_M
        x1 = (math.floor(xmax / ZONE_M) + 1) * ZONE_M
        y1 = (math.floor(ymax / ZONE_M) + 1) * ZONE_M
        return cls(x0, y0, cell_m, round((x1 - x0) / cell_m), round((y1 - y0) / cell_m))

    def cell_centres(self):
        """(xs, ys) arrays of shape (ny, nx) with each cell centre in metres."""
        cols = self.origin_x + (np.arange(self.nx) + 0.5) * self.cell_m
        rows = self.origin_y + (np.arange(self.ny) + 0.5) * self.cell_m
        return np.meshgrid(cols, rows)

    def cell_index(self, xs, ys):
        """(flat cell index of the points inside, which points are inside)."""
        col = np.floor((np.asarray(xs, float) - self.origin_x) / self.cell_m).astype(int)
        row = np.floor((np.asarray(ys, float) - self.origin_y) / self.cell_m).astype(int)
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        return row[inside] * self.nx + col[inside], inside


@dataclass(frozen=True)
class Grid:
    spec: GridSpec
    dwell: np.ndarray  # person-seconds per cell
    passage: np.ndarray  # distinct track ids per cell
    dropped: int  # samples that fell outside the grid

    @classmethod
    def from_samples(cls, spec, track_ids, xs, ys, fps):
        """One sample = one track seen at (x, y) in one frame."""
        cell, inside = spec.cell_index(xs, ys)
        size = spec.ny * spec.nx
        dwell = np.bincount(cell, minlength=size) / fps
        pairs = np.unique(np.stack([cell, np.asarray(track_ids)[inside]]), axis=1)
        passage = np.bincount(pairs[0], minlength=size).astype(float)
        shape = (spec.ny, spec.nx)
        return cls(spec, dwell.reshape(shape), passage.reshape(shape), int((~inside).sum()))
