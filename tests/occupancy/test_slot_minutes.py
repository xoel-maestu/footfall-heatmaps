"""Occupancy of a slot (improvements phase 6, H4): identity-free, from dwell only.

Mean occupancy is the slot's recorded dwell (person-seconds, the whole grid)
over the seconds the source covered; the peak is the busiest minute. Both come
from one aggregate total per minute for the whole grid: k=5 hides cells, not
the floor's total, so the total includes the dwell of hidden cells (they hold
about a fifth of the dwell of site 071). Nothing per cell or per track is kept
per minute, and a minute total reads as 0 when fewer than k distinct ids made
it (R5).
"""

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.application.record_site import RecordSite
from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import (
    MINUTES,
    SLOT_S,
    SlotAccumulator,
    SlotGrid,
    SlotMinutes,
    merge_incomplete_minutes,
    slot_reading,
)
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)


def crowd(n, x=0.1, y=0.1, first=0):
    return np.arange(first, first + n), np.full(n, x), np.full(n, y)


def minutes(person_s, observed_s, incomplete=False):
    p = np.zeros(MINUTES)
    o = np.zeros(MINUTES)
    p[: len(person_s)], o[: len(observed_s)] = person_s, observed_s
    return SlotMinutes("cam", T0, 0.5, p, o, incomplete)


# slot_reading: mean and peak occupancy


def test_mean_occupancy_is_the_recorded_dwell_over_observed_seconds():
    shown = np.array([[600.0, 0, 0, 0], [0, 0, 0, 0]])  # a 300 person-s cell is hidden at k
    m = minutes([120.0] * 5 + [60.0] * 5, [60.0] * 10)  # 900 person-s over 600 s observed

    r = slot_reading(shown, m, incomplete=True)

    assert r["observed_s"] == 600.0
    assert r["mean_occupancy"] == pytest.approx(900 / 600)


def test_peak_is_the_busiest_minute_and_its_start():
    m = minutes([60.0, 300.0, 120.0], [60.0, 60.0, 60.0])

    r = slot_reading(np.full((2, 4), 1.0), m, incomplete=True)

    assert r["peak_occupancy"] == pytest.approx(5.0)
    assert r["peak_start"] == T0 + 60


def test_a_minute_observed_under_30_s_is_not_a_peak_candidate():
    # the last minute of a video that stops 10 s into it: 4 person-s over 10 s is
    # 0.4 people, but over so short a span it is noise, not the busiest minute
    m = minutes([60.0, 90.0], [60.0, 10.0])

    r = slot_reading(np.ones((2, 4)), m, incomplete=True)

    assert r["peak_occupancy"] == pytest.approx(1.0)
    assert r["observed_s"] == 70.0


def test_a_complete_slot_without_minutes_has_no_reading():
    # a slot recorded before the minute totals: its shown grid leaves out the
    # cells k hides, so it would read low under the whole-floor headline
    r = slot_reading(np.full((2, 4), 450.0), None, incomplete=False)

    assert r == {"mean_occupancy": None, "peak_occupancy": None, "peak_start": None,
                 "observed_s": None}  # fmt: skip


def test_an_incomplete_slot_without_minutes_has_no_reading():
    r = slot_reading(np.ones((2, 4)), None, incomplete=True)

    assert r == {"mean_occupancy": None, "peak_occupancy": None, "peak_start": None,
                 "observed_s": None}  # fmt: skip


# SlotAccumulator: per-minute totals over the shown cells only


def test_minutes_total_the_whole_grid_per_minute_and_count_observed_time():
    a = SlotAccumulator(SPEC, T0)
    ids, xs, ys = crowd(6)
    cells, _ = SPEC.cell_index(xs, ys)
    lone, _ = SPEC.cell_index(np.array([1.9]), np.array([0.9]))
    for s in range(120):  # 6 people for two minutes, one sample a second
        if s < 10:  # and one more id for 10 s in a cell under k: hidden, still counted
            a.add(np.append(ids, 99), np.append(cells, lone), 1.0, T0 + s)
        else:
            a.add(ids, cells, 1.0, T0 + s)
    a.add(np.array([99]), lone, 30.0, T0 + 200)  # that id alone in minute 3: under k

    m = a.minutes("cam")
    (dwell,) = [g for g in a.grids("cam") if g.layer == "dwell"]

    assert m.person_s[:4].tolist() == [370.0, 360.0, 0.0, 0.0]
    assert m.observed_s[:4].tolist() == [60.0, 60.0, 0.0, 30.0]
    assert dwell.grid.sum() == 720.0  # the map shows the cells at k only
    assert m.person_s.shape == (MINUTES,) and m.observed_s.shape == (MINUTES,)


def test_a_minute_made_by_fewer_than_k_ids_reads_as_zero():
    a = SlotAccumulator(SPEC, T0)
    ids, xs, ys = crowd(6)
    cells, _ = SPEC.cell_index(xs, ys)
    a.add(ids, cells, 1.0, T0)  # minute 0: 6 ids
    a.add(ids[:2], cells[:2], 1.0, T0 + 61)  # minute 1: 2 ids

    m = a.minutes("cam")

    assert m.person_s[:2].tolist() == [6.0, 0.0]
    assert m.observed_s[:2].tolist() == [1.0, 1.0]


def test_minutes_of_two_incomplete_passes_add_up():
    old = minutes([60.0, 30.0], [60.0, 60.0], incomplete=True)
    new = minutes([0, 0, 90.0], [0, 0, 60.0], incomplete=True)

    m = merge_incomplete_minutes(old, new)

    assert m.person_s[:3].tolist() == [60.0, 30.0, 90.0]
    assert m.observed_s[:3].tolist() == [60.0, 60.0, 60.0]
    assert m.incomplete


# The recorders write the minutes with the slot


def test_record_slots_writes_the_minutes_with_its_slot():
    store = SqliteSlotStore(":memory:")
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0)
    for s in range(180):
        r.add(T0 + s, *crowd(6), seconds=1.0)
    r.close()

    (m,) = store.read_minutes("cam", T0, T0 + SLOT_S)
    (dwell,) = [s for s in store.read("cam", T0, T0 + SLOT_S) if s.layer == "dwell"]

    assert m.person_s[:4].tolist() == [360.0, 360.0, 360.0, 0.0]
    assert m.observed_s.sum() == 180.0
    assert m.person_s.sum() == dwell.grid.sum()
    assert m.incomplete


def test_record_site_fuses_the_minutes_with_the_site_weights():
    weights = np.zeros((2, 2, 4))
    weights[:, 0, 0] = 0.5  # cell (0, 0) seen by both cameras, half each
    store = SqliteSlotStore(":memory:")
    r = RecordSite(store, "site", ["a", "b"], weights, SPEC, retention_days=90,
                   now=lambda: T0)  # fmt: skip
    for s in range(120):
        r.add("a", T0 + s, *crowd(6), seconds=1.0)
        r.add("b", T0 + s, *crowd(6), seconds=1.0)
    r.close()

    (m,) = store.read_minutes("site", T0, T0 + SLOT_S)
    (dwell,) = [s for s in store.read("site", T0, T0 + SLOT_S) if s.layer == "dwell"]

    assert m.person_s[:2].tolist() == [360.0, 360.0]  # .5 x 360 + .5 x 360
    assert m.observed_s[:2].tolist() == [60.0, 60.0]
    assert m.person_s.sum() == pytest.approx(dwell.grid.sum())


def site_minutes(feed):
    weights = np.zeros((2, 2, 4))
    weights[:, 0, 0] = 0.5  # cell (0, 0) seen by both cameras, half each
    store = SqliteSlotStore(":memory:")
    r = RecordSite(store, "site", ["a", "b"], weights, SPEC, retention_days=90,
                   now=lambda: T0)  # fmt: skip
    feed(r)
    r.close()
    (m,) = store.read_minutes("site", T0, T0 + SLOT_S)
    return m


def test_a_minute_with_one_camera_down_fuses_the_cameras_that_covered_it():
    def feed(r):
        for s in range(120):
            r.add("a", T0 + s, *crowd(6), seconds=1.0)
            if s < 60:  # b is down in minute 1
                r.add("b", T0 + s, *crowd(6), seconds=1.0)

    m = site_minutes(feed)

    # minute 1: a alone carries the cell, its weight renormalised to 1, not .5 x 360
    assert m.person_s[:2].tolist() == [360.0, 360.0]
    assert m.observed_s[:2].tolist() == [60.0, 60.0]  # time any camera covered


def test_a_camera_that_never_reported_in_the_slot_does_not_dilute_its_minutes():
    def feed(r):
        for s in range(60):
            r.add("a", T0 + s, *crowd(6), seconds=1.0)

    assert site_minutes(feed).person_s[0] == 360.0


def test_store_keeps_a_complete_slots_minutes_over_an_incomplete_pass():
    store = SqliteSlotStore(":memory:")
    store.write([], [minutes([60.0], [60.0], incomplete=False)])
    store.write([], [minutes([5.0], [5.0], incomplete=True)])

    (m,) = store.read_minutes("cam", T0, T0 + SLOT_S)

    assert m.person_s[0] == 60.0 and not m.incomplete


def test_store_merges_two_incomplete_passes_of_minutes():
    store = SqliteSlotStore(":memory:")
    store.write([], [minutes([60.0], [60.0], incomplete=True)])
    store.write([], [minutes([0.0, 30.0], [0.0, 60.0], incomplete=True)])

    (m,) = store.read_minutes("cam", T0, T0 + SLOT_S)

    assert m.person_s[:2].tolist() == [60.0, 30.0] and m.observed_s.sum() == 120.0


def test_store_clamps_a_minutes_observed_time_to_60_s():
    # two incomplete passes that overlap in time add up; no minute covers more than 60 s
    store = SqliteSlotStore(":memory:")
    store.write([], [minutes([60.0], [60.0], incomplete=True)])
    store.write([], [minutes([30.0], [40.0], incomplete=True)])

    (m,) = store.read_minutes("cam", T0, T0 + SLOT_S)

    assert m.observed_s[0] == 60.0 and m.person_s[0] == 90.0


def test_a_slots_grids_and_minutes_follow_one_write_rule():
    def slot(value, incomplete):
        grids = [SlotGrid("cam", T0, 0.5, layer, np.full((2, 4), value), 6, incomplete)
                 for layer in ("dwell", "passage")]  # fmt: skip
        return grids, [minutes([value], [60.0], incomplete=incomplete)]

    store = SqliteSlotStore(":memory:")
    store.write(*slot(10.0, False))
    store.write(*slot(7.0, True))  # an incomplete pass over a complete slot: dropped in both

    grids = store.read("cam", T0, T0 + SLOT_S)
    (m,) = store.read_minutes("cam", T0, T0 + SLOT_S)
    assert all(g.grid[0, 0] == 10.0 and not g.incomplete for g in grids)
    assert m.person_s[0] == 10.0 and not m.incomplete

    store.write(*slot(8.0, False))  # a complete pass replaces both

    grids = store.read("cam", T0, T0 + SLOT_S)
    (m,) = store.read_minutes("cam", T0, T0 + SLOT_S)
    assert all(g.grid[0, 0] == 8.0 for g in grids) and m.person_s[0] == 8.0
