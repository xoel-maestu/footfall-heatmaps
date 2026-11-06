"""15-minute slots: the only occupancy data that is persisted (spec 3, R4, R5).

A slot is one floor grid per layer over a UTC wall-clock quarter: dwell in
person-seconds, passage in distinct ephemeral ids per cell, and n_unique, the
distinct ids of the whole slot. Nothing per track outlives the slot.
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
    n_unique: int  # distinct ids in the slot
    incomplete: bool  # the source did not cover the whole slot


def suppress(dwell, passage, k=K_MIN_PEOPLE):
    """R5: both layers empty where passage (distinct people per cell) is below k."""
    hidden = np.asarray(passage) < k
    return np.where(hidden, 0.0, dwell), np.where(hidden, 0.0, passage)


class SlotStore(Protocol):
    def write(self, slots: Sequence[SlotGrid]) -> None:
        """Insert, or replace a slot with the same camera, start, cell and layer."""
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

    def grids(self, camera_id, k=K_MIN_PEOPLE):
        """Both layers, suppressed below k: what may be written."""
        shape = (self.spec.ny, self.spec.nx)
        passage = np.zeros(self.spec.ny * self.spec.nx)
        if self.pairs:
            np.add.at(passage, [c for _, c in self.pairs], 1.0)
        dwell, passage = suppress(self.dwell.reshape(shape), passage.reshape(shape), k)
        return [
            SlotGrid(
                camera_id,
                self.start_utc,
                self.spec.cell_m,
                layer,
                grid,
                len(self.ids),
                self.incomplete,
            )
            for layer, grid in zip(LAYERS, (dwell, passage), strict=True)
        ]
