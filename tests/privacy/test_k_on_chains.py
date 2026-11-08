"""R5 under fragmentation (improvements plan phase 6, H3): k=5 counted on
stitched chains instead of raw track ids.

A tracker that loses a person behind a shelf gives them a new id when they
come back, so one person can count as several "distinct people" in a cell.
With k on chains, the ids that the slot's stitching joins count once in the
k test; the stored values (dwell, passage in raw ids) do not change where a
cell is shown. Without a stitch, everything is as before, byte for byte.
"""

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.application.record_site import RecordSite
from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec
from space_usage_heatmaps.occupancy.domain.slots import (
    K_MIN_PEOPLE,
    SLOT_S,
    SlotAccumulator,
    suppress,
)
from space_usage_heatmaps.occupancy.domain.stitching import K_STITCH, Stitch
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)
FRAME_S = 0.1  # 10 fps


def fragmented():
    """Frames of 2 people standing in cell (0, 0), each split into 3 ids by
    1 s gaps: 6 raw ids, 2 people. Yields (t, ids, xs, ys)."""
    pieces = [(0, 20), (30, 50), (60, 80)]
    for f in range(80):
        k = next((i for i, (a, b) in enumerate(pieces) if a <= f < b), None)
        if k is None:
            yield T0 + f * FRAME_S, np.empty(0, int), np.empty(0), np.empty(0)
            continue
        yield T0 + f * FRAME_S, np.array([1 + k, 4 + k]), np.array([0.1, 0.3]), np.array([0.1, 0.3])


def five_people():
    """5 people side by side in cell (0, 1), each one id the whole time."""
    for f in range(80):
        yield (
            T0 + f * FRAME_S,
            np.arange(10, 15),
            np.full(5, 0.6) + np.arange(5) * 0.05,
            np.full(5, 0.2),
        )


def record(stitch, *scenes):
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0, stitch=stitch)
    for frames in zip(*scenes, strict=True):
        t = frames[0][0]
        ids = np.concatenate([f[1] for f in frames])
        xs = np.concatenate([f[2] for f in frames])
        ys = np.concatenate([f[3] for f in frames])
        r.add(t, ids, xs, ys, FRAME_S)
    r.close()
    return {s.layer: s for s in store.read("cam", T0, T0 + SLOT_S)}


def test_two_people_split_into_six_ids_are_shown_with_raw_ids():
    rows = record(None, fragmented())

    assert rows["passage"].grid[0, 0] == 6
    assert rows["dwell"].grid[0, 0] > 0
    assert rows["dwell"].n_unique == 6


def test_two_people_split_into_six_ids_are_suppressed_with_chains():
    rows = record(K_STITCH, fragmented())

    assert rows["passage"].grid[0, 0] == 0
    assert rows["dwell"].grid[0, 0] == 0
    assert rows["dwell"].n_unique == 0  # 2 chains in the slot: below k


def test_five_people_who_never_fragment_stay_shown_with_chains():
    rows = record(K_STITCH, fragmented(), five_people())

    assert rows["passage"].grid[0, 1] == 5  # the stored value is the raw count
    assert rows["dwell"].grid[0, 1] == pytest.approx(5 * 80 * FRAME_S)
    assert rows["passage"].grid[0, 0] == 0  # the fragmented pair is still hidden
    assert rows["dwell"].n_unique == 11  # raw ids, shown: 7 chains reach k


def test_without_a_stitch_the_slot_is_byte_identical_to_raw_id_suppression():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 40, 3000)
    xs, ys = rng.uniform(-0.2, 2.2, 3000), rng.uniform(-0.2, 1.2, 3000)
    acc = SlotAccumulator(SPEC, T0)
    cells, inside = SPEC.cell_index(xs, ys)
    acc.add(ids[inside], cells, FRAME_S)
    acc.follow(ids, np.arange(3000), np.column_stack([xs, ys]), 1 / FRAME_S)  # no stitch: no-op

    g = Grid.from_samples(SPEC, ids, xs, ys, 1 / FRAME_S)
    raw = np.zeros(8)
    np.add.at(raw, cells, FRAME_S)  # as the slot sums dwell, sample by sample
    dwell, passage = suppress(raw.reshape(2, 4), g.passage)
    got = {s.layer: s for s in acc.grids("cam")}
    assert got["dwell"].grid.tobytes() == dwell.tobytes()
    assert got["passage"].grid.tobytes() == passage.tobytes()
    assert got["dwell"].n_unique == len(np.unique(ids[inside]))


def test_k_on_chains_applies_after_fusion_on_a_site():
    weights = np.zeros((2, 2, 4))
    weights[:, 0, 0] = 0.5
    weights[1, 0, 1] = 1.0
    out = {}
    for name, stitch in (("ids", None), ("chains", K_STITCH)):
        store = SqliteSlotStore(":memory:")
        r = RecordSite(store, "site", ["a", "b"], weights, SPEC, retention_days=90,
                       now=lambda: T0, stitch=stitch)  # fmt: skip
        for (t, ids, xs, ys), (_, ids2, xs2, ys2) in zip(fragmented(), fragmented(), strict=True):
            r.add("a", t, ids, xs, ys, FRAME_S)
            r.add("b", t, ids2, xs2, ys2, FRAME_S)
        r.close()
        out[name] = {s.layer: s for s in store.read("site", T0, T0 + SLOT_S)}

    assert out["ids"]["passage"].grid[0, 0] == pytest.approx(6.0)  # .5 x 6 + .5 x 6
    assert out["chains"]["passage"].grid[0, 0] == 0  # fused chains .5 x 2 + .5 x 2 < k
    assert out["chains"]["dwell"].grid[0, 0] == 0
    assert out["chains"]["dwell"].n_unique == 0


def test_the_product_stitch_is_the_accepted_g10_r1():
    assert Stitch(gap_s=10.0, radius_m=1.0, v_max_mps=1.5) == K_STITCH
    assert K_MIN_PEOPLE == 5
