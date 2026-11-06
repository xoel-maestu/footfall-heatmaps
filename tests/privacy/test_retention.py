"""R7 (spec 3): slots older than the retention (90 days by default) are purged."""

import numpy as np

from space_usage_heatmaps.occupancy.application.purge_slots import RETENTION_DAYS, purge_slots
from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import SlotGrid
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

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
