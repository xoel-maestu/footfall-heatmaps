"""15-minute slots: the only occupancy data that is persisted (spec 3, R4, R5).

A slot is one floor grid per layer over a UTC wall-clock quarter: dwell in
person-seconds, passage in distinct ephemeral ids per cell, and n_unique, the
distinct ids of the whole slot, 0 below k. Nothing per track outlives the slot.

With it go the slot's minute totals (SlotMinutes, improvements phase 6, H4):
per minute, two numbers for the whole grid, its dwell and the seconds of
source time the minute covered. They give the slot's occupancy without any id
count: mean occupancy is the recorded dwell over the observed seconds (people
in view on average), the peak is the busiest minute. k=5 hides cells of the
map; a total over the whole floor places no one, so it includes the dwell of
hidden cells (about a fifth of it on site 071, where the shown cells alone
read 24 % low). A minute total reads as 0 when fewer than k distinct ids made it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SLOT_S = 900  # 15 minutes
K_MIN_PEOPLE = 5  # R5: a cell with fewer distinct people reads as empty
LAYERS = ("dwell", "passage")
MINUTE_S = 60
MINUTES = SLOT_S // MINUTE_S
PEAK_MIN_OBSERVED_S = 30  # a minute the source covered less than this is no peak candidate


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


@dataclass(frozen=True)
class SlotMinutes:
    camera_id: str
    start_utc: int
    cell_m: float
    person_s: np.ndarray  # (MINUTES,) dwell of the whole grid per minute; 0 below k ids
    observed_s: np.ndarray  # (MINUTES,) seconds of source time each minute covered
    incomplete: bool


def minute_of(t, start_utc):
    """The minute of the slot that holds t, 0 to MINUTES - 1."""
    return min(max(int((t - start_utc) // MINUTE_S), 0), MINUTES - 1)


def merge_incomplete_minutes(old, new):
    """Two incomplete passes over one quarter cover disjoint time: both add up."""
    return SlotMinutes(
        new.camera_id,
        new.start_utc,
        new.cell_m,
        np.asarray(old.person_s) + np.asarray(new.person_s),
        np.asarray(old.observed_s) + np.asarray(new.observed_s),
        True,
    )


def slot_reading(dwell, minutes, incomplete):
    """The slot's occupancy from its minute totals: mean (people in view on
    average: the minutes' dwell over their observed seconds), the peak
    minute's occupancy and start, and the observed seconds. Without minutes (a
    slot recorded before them) a complete slot reads its shown dwell grid over
    the whole quarter, an incomplete one has none."""
    total = float(np.asarray(dwell).sum())
    out = {"mean_occupancy": None, "peak_occupancy": None, "peak_start": None, "observed_s": None}
    if minutes is None:
        if not incomplete:
            out |= {"mean_occupancy": total / SLOT_S, "observed_s": float(SLOT_S)}
        return out
    observed = np.asarray(minutes.observed_s, float)
    if observed.sum() <= 0:
        return out
    person_s = float(np.asarray(minutes.person_s, float).sum())
    out |= {"mean_occupancy": person_s / float(observed.sum()), "observed_s": float(observed.sum())}
    ok = observed >= PEAK_MIN_OBSERVED_S
    if ok.any():
        rate = np.where(ok, np.asarray(minutes.person_s, float) / np.where(ok, observed, 1), -1)
        m = int(rate.argmax())
        out |= {"peak_occupancy": float(rate[m]), "peak_start": minutes.start_utc + m * MINUTE_S}
    return out


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
    def write(self, slots: Sequence[SlotGrid], minutes: Sequence[SlotMinutes] = ()) -> None:
        """Insert a slot; on the same camera, start, cell and layer, a complete
        one replaces the stored one, an incomplete one is merged into a stored
        incomplete one (merge_incomplete) and dropped over a complete one.
        A slot's minute totals follow the same rule."""
        ...

    def read_minutes(self, camera_id: str, start_utc: int, end_utc: int) -> list[SlotMinutes]:
        """Minute totals of the slots starting in [start_utc, end_utc), by start."""
        ...

    def read(self, camera_id: str, start_utc: int, end_utc: int) -> list[SlotGrid]:
        """Slots starting in [start_utc, end_utc), by start then layer."""
        ...

    def purge(self, before_utc: int) -> int:
        """Delete slots and their minute totals starting before before_utc; how
        many slot rows went."""
        ...


class SlotAccumulator:
    """One open slot, in memory: dwell per cell, and the (id, cell) pairs and
    ids it has seen, needed for passage, k and n_unique and dropped on close.
    Per minute too, in memory only: dwell per cell (a site fuses it per cell),
    (id, cell) pairs and the seconds of source time."""

    def __init__(self, spec, start_utc, incomplete=False):
        self.spec, self.start_utc, self.incomplete = spec, start_utc, incomplete
        self.dwell = np.zeros(spec.ny * spec.nx)
        self.pairs = set()
        self.ids = set()
        self.minute_dwell = np.zeros((MINUTES, spec.ny * spec.nx))
        self.minute_pairs = [set() for _ in range(MINUTES)]
        self.observed = np.zeros(MINUTES)

    def add(self, track_ids, cells, seconds, t):
        """Samples of one frame at t, each worth seconds; the frame adds
        seconds of observed time to its minute, with or without anyone."""
        np.add.at(self.dwell, cells, seconds)
        ids = np.asarray(track_ids).tolist()
        pairs = list(zip(ids, np.asarray(cells).tolist(), strict=True))
        self.pairs.update(pairs)
        self.ids.update(ids)
        m = minute_of(t, self.start_utc)
        np.add.at(self.minute_dwell[m], cells, seconds)
        self.minute_pairs[m].update(pairs)
        self.observed[m] += seconds

    def minute_raw(self, shown):
        """Per minute, before k: dwell over the cells of a flat bool mask and
        the distinct ids that made it."""
        person_s = self.minute_dwell[:, shown].sum(axis=1)
        ids = [len({i for i, c in pairs if shown[c]}) for pairs in self.minute_pairs]
        return person_s, np.asarray(ids)

    def minutes(self, camera_id, k=K_MIN_PEOPLE):
        """The slot's minute totals over the whole grid, 0 below k ids: what may be written."""
        person_s, ids = self.minute_raw(np.ones(self.dwell.size, bool))
        return SlotMinutes(
            camera_id,
            self.start_utc,
            self.spec.cell_m,
            np.where(ids < k, 0.0, person_s),
            self.observed.copy(),
            self.incomplete,
        )

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
