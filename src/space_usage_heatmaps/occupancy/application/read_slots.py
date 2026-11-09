"""Use case: read persisted slots as they are published, k-suppressed (R5)."""

from dataclasses import replace

import numpy as np

from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, suppress, suppress_count


def read_slots(store, camera_id, start_utc, end_utc, k=K_MIN_PEOPLE):
    """Every layer of a slot is suppressed with its passage layer, and n_unique
    with k; a slot without a passage layer reads as empty."""
    slots = store.read(camera_id, start_utc, end_utc)
    passage = {(s.start_utc, s.cell_m): s.grid for s in slots if s.layer == "passage"}
    out = []
    for s in slots:
        people = passage.get((s.start_utc, s.cell_m), np.zeros_like(s.grid))
        dwell, people = suppress(s.grid, people, k)
        grid = people if s.layer == "passage" else dwell
        out.append(replace(s, grid=grid, n_unique=suppress_count(s.n_unique, k)))
    return out


def read_minutes(store, camera_id, start_utc, end_utc):
    """The slots' minute totals, as written: k already applied per minute
    over the cells the slot shows (SlotAccumulator.minutes, RecordSite)."""
    return store.read_minutes(camera_id, start_utc, end_utc)
