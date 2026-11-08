"""RecordSite: several cameras of one floor into fused 15-min slots (improvements phase 5).

Each camera keeps its own open slot in memory; when every camera has moved
past a quarter (or the sources end), the cameras' raw grids are fused with the
site's frozen weights, k=5 applies to the fused grid, and only that is written.
"""

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.application.record_site import RecordSite
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import SLOT_S
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)
# Cell (0, 0) is seen by both cameras, half each; cell (0, 3) by camera b only.
WEIGHTS = np.zeros((2, 2, 4))
WEIGHTS[:, 0, 0] = 0.5
WEIGHTS[1, 0, 3] = 1.0


def site(store, now=T0, k=5):
    return RecordSite(store, "site", ["a", "b"], WEIGHTS, SPEC, retention_days=90,
                      now=lambda: now, k=k)  # fmt: skip


def crowd(n, x=0.1, y=0.1, first=0):
    return np.arange(first, first + n), np.full(n, x), np.full(n, y)


def rows(store):
    return {s.layer: s for s in store.read("site", T0, T0 + 4 * SLOT_S)}


def test_the_fused_grid_is_the_weighted_cameras_and_k_applies_after_fusion():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)  # 6 people in cell (0, 0)
    r.add("b", T0, *crowd(4), seconds=1.0)  # 4 of them seen by b: fused 5 people
    r.close()

    got = rows(store)
    assert got["passage"].grid[0, 0] == pytest.approx(5.0)
    assert got["dwell"].grid[0, 0] == pytest.approx(5.0)  # .5 x 6 s + .5 x 4 s
    assert got["passage"].camera_id == "site"


def test_a_fused_cell_under_k_is_empty_even_where_one_camera_saw_k():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("b", T0, *crowd(2), seconds=1.0)  # fused 4 people: hidden
    r.close()

    got = rows(store)
    assert got["passage"].grid.sum() == 0 and got["dwell"].grid.sum() == 0


def test_cells_outside_every_camera_weight_stay_empty():
    store = SqliteSlotStore(":memory:")
    r = site(store, k=1)

    r.add("a", T0, *crowd(3, x=1.1), seconds=1.0)  # cell (0, 2): no camera weighs it
    r.add("b", T0, *crowd(3, x=1.6), seconds=1.0)  # cell (0, 3): b alone
    r.close()

    got = rows(store)
    assert got["dwell"].grid[0, 2] == 0
    assert got["dwell"].grid[0, 3] == pytest.approx(3.0)


def test_n_unique_is_the_most_any_one_camera_saw():
    """Ids are per camera: adding cameras would count one person twice."""
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("b", T0, *crowd(7, x=1.6), seconds=1.0)
    r.close()

    assert rows(store)["dwell"].n_unique == 7


def test_a_slot_closes_once_every_camera_moved_past_it():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("b", T0, *crowd(6), seconds=1.0)
    r.add("a", T0 + SLOT_S, *crowd(0), seconds=1.0)
    assert store.read("site", T0, T0 + SLOT_S) == []  # b is still in the first quarter
    r.add("b", T0 + SLOT_S + 1, *crowd(0), seconds=1.0)

    first = store.read("site", T0, T0 + SLOT_S)
    assert [(s.layer, s.incomplete) for s in first] == [("dwell", False), ("passage", False)]
    assert r.written == 1


def test_a_camera_that_never_reported_a_slot_makes_it_incomplete():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("a", T0 + SLOT_S, *crowd(0), seconds=1.0)
    r.add("b", T0 + SLOT_S, *crowd(0), seconds=1.0)

    assert {s.incomplete for s in store.read("site", T0, T0 + SLOT_S)} == {True}


def test_a_camera_drop_marks_the_open_slot_incomplete():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("b", T0, *crowd(6), seconds=1.0)
    r.mark_incomplete("b")
    r.add("a", T0 + SLOT_S, *crowd(0), seconds=1.0)
    r.add("b", T0 + SLOT_S, *crowd(0), seconds=1.0)

    assert {s.incomplete for s in store.read("site", T0, T0 + SLOT_S)} == {True}


def test_a_slot_past_the_retention_is_not_written():
    store = SqliteSlotStore(":memory:")
    r = site(store, now=T0 + 100 * 86_400)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.close()

    assert store.read("site", 0, T0 + 4 * SLOT_S) == []
    assert r.past_retention == 1 and r.written == 0


def test_unknown_cameras_and_mismatched_weights_are_errors():
    store = SqliteSlotStore(":memory:")
    with pytest.raises(ValueError, match="weights"):
        RecordSite(store, "site", ["a"], WEIGHTS, SPEC)
    r = site(store)
    with pytest.raises(KeyError):
        r.add("c", T0, *crowd(1), seconds=1.0)


def test_the_live_layer_is_fused_and_k_suppressed():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("b", T0, *crowd(4), seconds=1.0)
    dwell, passage = r.live(T0)
    assert dwell[0, 0] == pytest.approx(5.0) and passage[0, 0] == pytest.approx(5.0)

    r2 = site(SqliteSlotStore(":memory:"))
    r2.add("a", T0, *crowd(6), seconds=1.0)
    r2.add("b", T0, *crowd(2), seconds=1.0)
    dwell, passage = r2.live(T0)
    assert dwell.sum() == 0 and passage.sum() == 0


def test_nothing_per_track_survives_a_written_slot():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("b", T0, *crowd(6), seconds=1.0)
    r.add("a", T0 + SLOT_S, *crowd(0), seconds=1.0)
    r.add("b", T0 + SLOT_S, *crowd(0), seconds=1.0)

    assert r.open_starts() == [T0 + SLOT_S]  # the first quarter's ids are gone


def test_a_camera_that_stays_down_does_not_hold_the_slots_open():
    store = SqliteSlotStore(":memory:")
    r = site(store)

    r.add("a", T0, *crowd(6), seconds=1.0)
    r.add("a", T0 + SLOT_S, *crowd(0), seconds=1.0)
    assert r.written == 0
    r.add("a", T0 + 2 * SLOT_S, *crowd(0), seconds=1.0)

    assert r.written == 1 and r.open_starts() == [T0 + SLOT_S, T0 + 2 * SLOT_S]


def test_samples_for_a_quarter_already_written_are_dropped():
    store = SqliteSlotStore(":memory:")
    r = site(store)
    for c in ("a", "b"):
        r.add(c, T0, *crowd(6), seconds=1.0)
    for c in ("a", "b"):
        r.add(c, T0 + SLOT_S, *crowd(0), seconds=1.0)

    r.add("b", T0 + 10, *crowd(3), seconds=1.0)  # a clock that stepped back

    assert r.late == 3 and r.open_starts() == [T0 + SLOT_S]
