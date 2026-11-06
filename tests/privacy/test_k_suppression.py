"""R5 (spec 3): a cell of a persisted slot with fewer than k=5 distinct people reads as empty.

The passage layer is the per-cell count of distinct ephemeral ids, an
aggregate R4 allows; suppression runs before the write, so no cell below k
reaches the database, and the read path applies it again from that layer.
"""

import numpy as np

from space_usage_heatmaps.occupancy.application.read_slots import read_slots
from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, SlotGrid, suppress
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

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
