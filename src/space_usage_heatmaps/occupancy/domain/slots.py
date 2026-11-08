"""15-minute slots: the only occupancy data that is persisted (spec 3, R4, R5).

A slot is one floor grid per layer over a UTC wall-clock quarter: dwell in
person-seconds, passage in distinct ephemeral ids per cell, and n_unique, the
distinct ids of the whole slot, 0 below k. Nothing per track outlives the slot.

The k test counts raw track ids, or with a stitch (phase 6, H3) the slot's
stitched chains: ids the stitch joins count once, so one person the tracker
split into several ids does not pass k alone. The stored values stay the raw
ids' where a cell is shown.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from space_usage_heatmaps.occupancy.domain.stitching import ChainBuilder

SLOT_S = 900  # 15 minutes
K_MIN_PEOPLE = 5  # R5: a cell with fewer distinct people reads as empty
LAYERS = ("dwell", "passage")


def follow_rows(slot, t, track_ids, xs, ys, seconds):
    """A recorder's frame at t into the slot's stitch: frames counted from the
    slot start in steps of seconds (each sample's dwell), so the stitch's gap
    and window are in the source's own time."""
    if slot.stitch is None:
        return
    n = len(np.asarray(track_ids))
    frames = np.full(n, round((t - slot.start_utc) / seconds), dtype=np.int64)
    slot.follow(track_ids, frames, np.column_stack([xs, ys]).reshape(-1, 2), 1 / seconds)


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


def suppress(dwell, passage, k=K_MIN_PEOPLE, people=None):
    """R5: both layers empty where passage (distinct people per cell) is below
    k; people: the per-cell count k reads instead of passage (chains)."""
    hidden = np.asarray(passage if people is None else people) < k
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
    ids it has seen, needed for passage, k and n_unique and dropped on close.
    With a stitch, also the chain builder of its tracks (follow), for k."""

    def __init__(self, spec, start_utc, incomplete=False, stitch=None):
        self.spec, self.start_utc, self.incomplete = spec, start_utc, incomplete
        self.dwell = np.zeros(spec.ny * spec.nx)
        self.pairs = set()
        self.ids = set()
        self.stitch = stitch
        self._chains = None  # ChainBuilder, made on the first follow

    def add(self, track_ids, cells, seconds):
        np.add.at(self.dwell, cells, seconds)
        ids = np.asarray(track_ids).tolist()
        self.pairs.update(zip(ids, np.asarray(cells).tolist(), strict=True))
        self.ids.update(ids)

    def follow(self, track_ids, frames, xy, fps):
        """Every row of the slot's tracks, inside the grid or not (xy NaN where
        not placed), integer frames at fps: what the stitch reads. No-op
        without a stitch."""
        if self.stitch is None:
            return
        if self._chains is None:
            self._chains = ChainBuilder(self.stitch, fps)
        self._chains.add(track_ids, frames, xy)

    def raw(self):
        """(dwell, passage) before k: for fusion in memory, never written as is."""
        shape = (self.spec.ny, self.spec.nx)
        passage = np.zeros(self.spec.ny * self.spec.nx)
        if self.pairs:
            np.add.at(passage, [c for _, c in self.pairs], 1.0)
        return self.dwell.reshape(shape), passage.reshape(shape)

    def _chain_of(self):
        return {} if self._chains is None else self._chains.chains()

    def people(self):
        """(per-cell count, slot count) that k reads: distinct raw ids, or with
        a stitch distinct chains. None, None without a stitch (k reads passage)."""
        if self.stitch is None:
            return None, None
        chain = self._chain_of()
        pairs = {(chain.get(i, i), c) for i, c in self.pairs}
        people = np.zeros(self.spec.ny * self.spec.nx)
        if pairs:
            np.add.at(people, [c for _, c in pairs], 1.0)
        n = len({chain.get(i, i) for i in self.ids})
        return people.reshape(self.spec.ny, self.spec.nx), n

    def grids(self, camera_id, k=K_MIN_PEOPLE):
        """Both layers, suppressed below k: what may be written."""
        people, n_people = self.people()
        dwell, passage = suppress(*self.raw(), k, people)
        n = len(self.ids) if n_people is None or n_people >= k else 0
        return [
            SlotGrid(
                camera_id,
                self.start_utc,
                self.spec.cell_m,
                layer,
                grid,
                suppress_count(n, k),
                self.incomplete,
            )
            for layer, grid in zip(LAYERS, (dwell, passage), strict=True)
        ]
