"""Privacy rule R7: slots older than the retention (90 days by default) are purged."""

import numpy as np

from footfall_heatmaps.occupancy.application.purge_slots import RETENTION_DAYS, purge_slots
from footfall_heatmaps.occupancy.application.record_slots import RecordSlots
from footfall_heatmaps.occupancy.domain.grid import GridSpec
from footfall_heatmaps.occupancy.domain.slots import SlotGrid
from footfall_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

DAY = 86_400
NOW = 1_793_955_600
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=2, ny=2)


def slot(start):
    return SlotGrid("cam", start, 0.5, "passage", np.zeros((2, 2)), 0, False)


def test_the_default_is_90_days():
    assert RETENTION_DAYS == 90


def test_purge_drops_slots_that_ended_before_the_retention_window():
    store = SqliteSlotStore(":memory:")
    old, kept = NOW - 90 * DAY - 900, NOW - 90 * DAY
    store.write([slot(old), slot(kept), slot(NOW)])

    assert purge_slots(store, now=NOW) == 1
    assert [s.start_utc for s in store.read("cam", 0, NOW + 900)] == [kept, NOW]


def test_the_retention_is_configurable():
    store = SqliteSlotStore(":memory:")
    store.write([slot(NOW - 2 * DAY), slot(NOW)])

    assert purge_slots(store, now=NOW, retention_days=1) == 1


def test_recording_purges_as_it_writes():
    store = SqliteSlotStore(":memory:")
    store.write([slot(NOW - 10 * DAY)])
    r = RecordSlots(store, "cam", SPEC, retention_days=7, now=lambda: NOW)

    r.add(NOW, np.arange(5), np.full(5, 0.1), np.full(5, 0.1), seconds=1.0)
    r.close()

    assert {s.start_utc for s in store.read("cam", 0, NOW + 900)} == {NOW}


def test_a_slot_already_past_the_retention_is_not_counted_as_written():
    """Regression: a video recorded with an old --start wrote each slot and purged
    it at once, while the run reported it as written."""
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: NOW)

    for i in range(10):
        r.add(NOW - 200 * DAY + i, np.arange(6), np.full(6, 0.1), np.full(6, 0.1), seconds=1.0)
    r.close()

    assert (r.written, r.past_retention) == (0, 1)
    assert store.read("cam", 0, NOW + 900) == []


def test_a_slot_inside_the_retention_is_written_and_not_reported():
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: NOW)

    r.add(NOW - 89 * DAY, np.arange(6), np.full(6, 0.1), np.full(6, 0.1), seconds=1.0)
    r.close()

    assert (r.written, r.past_retention) == (1, 0)


def test_purge_drops_the_minute_totals_of_the_slots_it_drops():
    store = SqliteSlotStore(":memory:")
    old = NOW - 90 * DAY - 900
    for start in (old, NOW):
        r = RecordSlots(store, "cam", SPEC, retention_days=365, now=lambda: NOW)
        r.add(start, np.arange(6), np.full(6, 0.1), np.full(6, 0.1), seconds=1.0)
        r.close()
    assert len(store.read_minutes("cam", 0, NOW + 900)) == 2

    purge_slots(store, now=NOW)

    assert [m.start_utc for m in store.read_minutes("cam", 0, NOW + 900)] == [NOW]
