"""Use case: read persisted slots as they are published, k-suppressed (R5)."""

from dataclasses import replace

import numpy as np

from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, suppress


def read_slots(store, camera_id, start_utc, end_utc, k=K_MIN_PEOPLE):
    """Every layer of a slot is suppressed with its passage layer; a slot without
    one reads as empty."""
    slots = store.read(camera_id, start_utc, end_utc)
    passage = {(s.start_utc, s.cell_m): s.grid for s in slots if s.layer == "passage"}
    out = []
    for s in slots:
        people = passage.get((s.start_utc, s.cell_m), np.zeros_like(s.grid))
        dwell, people = suppress(s.grid, people, k)
        grid = people if s.layer == "passage" else dwell
        out.append(replace(s, grid=grid))
    return out
