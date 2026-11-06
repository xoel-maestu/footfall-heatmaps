import numpy as np

from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import SLOT_S, slot_start
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

T0 = 1_793_955_600  # 2026-11-06T09:00:00Z, a quarter
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)


def recorder(store, now=T0):
    return RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: now)


def crowd(n, x=0.1, y=0.1, first=0):
    """n tracks, all in the same cell."""
    return np.arange(first, first + n), np.full(n, x), np.full(n, y)


def test_slots_are_aligned_to_utc_quarters():
    assert slot_start(T0) == T0
    assert slot_start(T0 + 899.9) == T0
    assert slot_start(T0 + 900) == T0 + SLOT_S


def test_a_slot_is_written_when_time_crosses_into_the_next_one():
    store = SqliteSlotStore(":memory:")
    r = recorder(store)

    r.add(T0, *crowd(6), seconds=1.0)
    assert store.read("cam", T0, T0 + 2 * SLOT_S) == []
    r.add(T0 + SLOT_S, *crowd(0), seconds=1.0)

    rows = store.read("cam", T0, T0 + 2 * SLOT_S)
    assert [(s.start_utc, s.layer, s.n_unique, s.incomplete) for s in rows] == [
        (T0, "dwell", 6, False),
        (T0, "passage", 6, False),
    ]
    assert rows[0].grid.shape == (2, 4) and rows[0].cell_m == 0.5
    assert rows[0].grid[0, 0] == 6.0  # six tracks, one second each
    assert rows[1].grid[0, 0] == 6.0  # six distinct ids


def test_the_slot_open_when_the_source_ends_is_incomplete():
    store = SqliteSlotStore(":memory:")
    r = recorder(store)

    r.add(T0, *crowd(5), seconds=1.0)
    r.close()

    assert {s.incomplete for s in store.read("cam", T0, T0 + SLOT_S)} == {True}


def test_a_slot_the_source_joins_late_is_incomplete():
    store = SqliteSlotStore(":memory:")
    r = recorder(store)

    r.add(T0 + 600, *crowd(5), seconds=1 / 30)
    r.add(T0 + SLOT_S, *crowd(5), seconds=1 / 30)
    r.close()

    rows = store.read("cam", T0, T0 + 2 * SLOT_S)
    assert [(s.start_utc, s.incomplete) for s in rows if s.layer == "dwell"] == [
        (T0, True),
        (T0 + SLOT_S, True),  # still open at the end
    ]


def test_a_drop_marks_the_open_slot_incomplete_only():
    store = SqliteSlotStore(":memory:")
    r = recorder(store)

    r.add(T0, *crowd(5), seconds=1.0)
    r.mark_incomplete()
    r.add(T0 + SLOT_S, *crowd(5), seconds=1.0)
    r.add(T0 + 2 * SLOT_S, *crowd(0), seconds=1.0)

    rows = store.read("cam", T0, T0 + 2 * SLOT_S)
    assert [(s.start_utc, s.incomplete) for s in rows if s.layer == "dwell"] == [
        (T0, True),
        (T0 + SLOT_S, False),
    ]


def test_n_unique_counts_distinct_ids_over_the_slot():
    store = SqliteSlotStore(":memory:")
    r = recorder(store)

    ids, xs = np.array([1, 2, 3, 4, 5, 6]), np.array([0.1, 0.6, 1.1, 0.1, 0.6, 1.1])
    for _ in range(10):  # the same six ids, frame after frame
        r.add(T0, ids, xs, np.full(6, 0.1), seconds=0.1)
    r.close()

    assert {s.n_unique for s in store.read("cam", T0, T0 + SLOT_S)} == {6}


def test_live_layer_halves_every_60_seconds():
    store = SqliteSlotStore(":memory:")
    r = recorder(store)

    r.add(T0, *crowd(6), seconds=1.0)
    dwell, passage = r.live(T0 + 60)

    assert dwell[0, 0] == 3.0  # 6 person-seconds, one half-life later
    assert passage[0, 0] == 3.0  # 6 entries


def test_live_passage_counts_a_track_entering_a_cell_once():
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0, k=1)  # one track shown

    for t in range(3):  # track 0 stays in cell (0, 0)
        r.add(T0 + t * 1e-6, np.array([0]), np.array([0.1]), np.array([0.1]), seconds=0.1)
    r.add(T0 + 3e-6, np.array([0]), np.array([0.6]), np.array([0.1]), seconds=0.1)

    _, passage = r.live(T0 + 3e-6)
    np.testing.assert_allclose(passage[0, :2], [1.0, 1.0], rtol=1e-6)
