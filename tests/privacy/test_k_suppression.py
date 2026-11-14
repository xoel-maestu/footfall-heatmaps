"""Privacy rule R5: a cell with fewer than k=5 distinct people reads as empty, in a
persisted slot and, by the same rule, in the live layer.

The passage layer is the per-cell count of distinct ephemeral ids, an
aggregate R4 allows; suppression runs before the write, so no cell below k
reaches the database, and the read path applies it again from that layer.
"""

import numpy as np

from footfall_heatmaps.occupancy.application.read_slots import read_slots
from footfall_heatmaps.occupancy.application.record_slots import RecordSlots
from footfall_heatmaps.occupancy.domain.grid import GridSpec
from footfall_heatmaps.occupancy.domain.live import RECENT_S, LiveLayer
from footfall_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, SlotGrid, suppress
from footfall_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

T0 = 1_793_955_600
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)


def record(store):
    """Cell (0, 0): 5 people; cell (0, 1): 4 people, staying much longer."""
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0)
    ids = np.arange(9)
    xs = np.r_[np.full(5, 0.1), np.full(4, 0.6)]
    for _ in range(100):
        r.add(T0, ids, xs, np.full(9, 0.1), seconds=1.0)
    r.close()


def test_k_is_five():
    assert K_MIN_PEOPLE == 5


def test_suppress_empties_both_layers_below_k():
    dwell = np.array([[10.0, 50.0, 0.0]])
    passage = np.array([[5.0, 4.0, 0.0]])

    d, p = suppress(dwell, passage)

    np.testing.assert_array_equal(d, [[10.0, 0.0, 0.0]])
    np.testing.assert_array_equal(p, [[5.0, 0.0, 0.0]])


def test_no_cell_below_k_is_written():
    store = SqliteSlotStore(":memory:")
    record(store)

    rows = {s.layer: s.grid for s in store.read("cam", T0, T0 + 900)}

    assert rows["passage"][0, 0] == 5 and rows["dwell"][0, 0] == 500
    assert rows["passage"][0, 1] == 0 and rows["dwell"][0, 1] == 0
    assert not ((rows["passage"] > 0) & (rows["passage"] < K_MIN_PEOPLE)).any()


def test_the_read_path_empties_cells_below_k_even_if_stored():
    """A row written by anything else (an older run, a bug) is still suppressed on read."""
    store = SqliteSlotStore(":memory:")
    passage = np.array([[5.0, 4.0, 1.0, 0.0], [0.0] * 4])
    dwell = np.array([[9.0, 8.0, 7.0, 0.0], [0.0] * 4])
    store.write(
        [
            SlotGrid("cam", T0, 0.5, "dwell", dwell, 10, False),
            SlotGrid("cam", T0, 0.5, "passage", passage, 10, False),
        ]
    )

    out = {s.layer: s.grid for s in read_slots(store, "cam", T0, T0 + 900)}

    np.testing.assert_array_equal(out["dwell"][0], [9.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["passage"][0], [5.0, 0.0, 0.0, 0.0])


def test_a_dwell_row_without_its_passage_row_reads_empty():
    store = SqliteSlotStore(":memory:")
    store.write([SlotGrid("cam", T0, 0.5, "dwell", np.full((2, 4), 9.0), 10, False)])

    (out,) = read_slots(store, "cam", T0, T0 + 900)

    assert not out.grid.any()


def test_n_unique_below_k_is_written_as_zero():
    """A slot with four people would otherwise say "four people were here" beside empty grids."""
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0)
    r.add(T0, np.arange(4), np.full(4, 0.1), np.full(4, 0.1), seconds=1.0)
    r.close()

    assert {s.n_unique for s in store.read("cam", T0, T0 + 900)} == {0}


def test_n_unique_at_k_is_written():
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0)
    r.add(T0, np.arange(5), np.full(5, 0.1), np.full(5, 0.1), seconds=1.0)
    r.close()

    assert {s.n_unique for s in store.read("cam", T0, T0 + 900)} == {5}


def test_the_read_path_empties_n_unique_below_k_even_if_stored():
    store = SqliteSlotStore(":memory:")
    empty = np.zeros((2, 4))
    store.write(
        [
            SlotGrid("cam", T0, 0.5, "dwell", empty, 3, False),
            SlotGrid("cam", T0, 0.5, "passage", empty, 3, False),
        ]
    )

    assert {s.n_unique for s in read_slots(store, "cam", T0, T0 + 900)} == {0}


# The live layer: a cell shows only once k distinct ids were seen in it within
# the last two half-lives (120 s); the (id, cell, last seen) bookkeeping is in
# memory and pruned as it ages.


def live_cell(n, t, first=0, layer=None):
    """n ids in cell (0, 0) at t, one second of dwell each."""
    layer = layer or LiveLayer((2, 4))
    layer.add(t, np.arange(first, first + n), np.zeros(n, dtype=int), 1.0)
    return layer


def test_live_cell_with_four_people_reads_as_empty():
    dwell, passage = live_cell(4, T0).at(T0)

    assert not dwell.any() and not passage.any()


def test_live_cell_with_five_people_is_shown():
    dwell, passage = live_cell(5, T0).at(T0)

    assert dwell[0, 0] == 5.0 and passage[0, 0] == 5.0


def test_live_ids_age_out_of_the_recent_window():
    layer = live_cell(5, T0)
    assert RECENT_S == 120

    assert layer.at(T0 + 119)[0][0, 0] > 0
    dwell, passage = layer.at(T0 + 121)  # decayed but still above zero, and hidden
    assert not dwell.any() and not passage.any()
    assert layer.last_seen == {}  # pruned, not only ignored


def test_live_window_counts_people_seen_at_different_times():
    layer = live_cell(4, T0)
    live_cell(1, T0 + 100, first=4, layer=layer)

    assert layer.at(T0 + 100)[1][0, 0] > 0  # 5 distinct ids within 120 s
    assert not layer.at(T0 + 121)[1].any()  # the first 4 aged out: 1 left
