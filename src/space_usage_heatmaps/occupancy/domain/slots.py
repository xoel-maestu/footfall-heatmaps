"""15-minute slots: the only occupancy data that is persisted (spec 3, R4, R5).

A slot is one floor grid per layer over a UTC wall-clock quarter: dwell in
person-seconds, passage in distinct ephemeral ids per cell, and n_unique, the
distinct ids of the whole slot, 0 below k. Nothing per track outlives the slot.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SLOT_S = 900  # 15 minutes
K_MIN_PEOPLE = 5  # R5: a cell with fewer distinct people reads as empty
LAYERS = ("dwell", "passage")


def slot_start(t):
    """The quarter that holds t (UTC epoch seconds), as epoch seconds."""
    return int(t // SLOT_S * SLOT_S)


@dataclass(frozen=True)
class SlotGrid:
    camera_id: str
    start_utc: int  # epoch seconds, a multiple of SLOT_S
    cell_m: float
    layer: str  # one of LAYERS
    grid: np.ndarray  # (ny, nx) floats
    n_unique: int  # distinct ids in the slot; 0 below k (R5)
    incomplete: bool  # the source did not cover the whole slot


def suppress(dwell, passage, k=K_MIN_PEOPLE):
    """R5: both layers empty where passage (distinct people per cell) is below k."""
    hidden = np.asarray(passage) < k
    return np.where(hidden, 0.0, dwell), np.where(hidden, 0.0, passage)


def suppress_count(n, k=K_MIN_PEOPLE):
    """R5: a slot's count of distinct people reads as 0 below k."""
    return n if n >= k else 0


def merge_incomplete(old, new, k=K_MIN_PEOPLE):
    """Two incomplete passes over one quarter (a restart): ids differ across
    runs, so dwell adds up (disjoint time) and passage and n_unique take the
    larger pass, which never counts one person twice. Inputs are already
    suppressed, so a passage cell is 0 or at least k, and so is their max."""
    if old.layer == "dwell":
        grid = np.asarray(old.grid) + np.asarray(new.grid)
    else:
        grid = np.where(np.maximum(old.grid, new.grid) < k, 0.0, np.maximum(old.grid, new.grid))
    n = suppress_count(max(old.n_unique, new.n_unique), k)
    return SlotGrid(new.camera_id, new.start_utc, new.cell_m, new.layer, grid, n, True)


class SlotStore(Protocol):
    def write(self, slots: Sequence[SlotGrid]) -> None:
        """Insert a slot; on the same camera, start, cell and layer, a complete
        one replaces the stored one, an incomplete one is merged into a stored
        incomplete one (merge_incomplete) and dropped over a complete one."""
        ...

    def read(self, camera_id: str, start_utc: int, end_utc: int) -> list[SlotGrid]:
        """Slots starting in [start_utc, end_utc), by start then layer."""
        ...

    def purge(self, before_utc: int) -> int:
        """Delete slots starting before before_utc; how many rows went."""
        ...


class SlotAccumulator:
    """One open slot, in memory: dwell per cell, and the (id, cell) pairs and
    ids it has seen, needed for passage, k and n_unique and dropped on close."""

    def __init__(self, spec, start_utc, incomplete=False):
        self.spec, self.start_utc, self.incomplete = spec, start_utc, incomplete
        self.dwell = np.zeros(spec.ny * spec.nx)
        self.pairs = set()
        self.ids = set()

    def add(self, track_ids, cells, seconds):
        np.add.at(self.dwell, cells, seconds)
        ids = np.asarray(track_ids).tolist()
        self.pairs.update(zip(ids, np.asarray(cells).tolist(), strict=True))
        self.ids.update(ids)

    def raw(self):
        """(dwell, passage) before k: for fusion in memory, never written as is."""
        shape = (self.spec.ny, self.spec.nx)
        passage = np.zeros(self.spec.ny * self.spec.nx)
        if self.pairs:
            np.add.at(passage, [c for _, c in self.pairs], 1.0)
        return self.dwell.reshape(shape), passage.reshape(shape)

    def grids(self, camera_id, k=K_MIN_PEOPLE):
        """Both layers, suppressed below k: what may be written."""
        dwell, passage = suppress(*self.raw(), k)
        return [
            SlotGrid(
                camera_id,
                self.start_utc,
                self.spec.cell_m,
                layer,
                grid,
                suppress_count(len(self.ids), k),
                self.incomplete,
            )
            for layer, grid in zip(LAYERS, (dwell, passage), strict=True)
        ]
