"""R4 (spec 3): only aggregated grids are persisted, never trajectories or images.

The database a run writes holds two tables with exactly the contract's columns:
slots, whose grid blob decodes to one floor grid of numbers, and slot_minutes,
one aggregate total per minute of a slot for the whole grid (person-seconds
and observed seconds; phase 6, H4). Nothing per track, nothing per cell per minute.
"""

import sqlite3

import numpy as np

from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import SlotGrid
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

T0 = 1_793_955_600
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)
COLUMNS = ["camera_id", "start_utc", "cell_m", "layer", "grid_blob", "n_unique", "incomplete"]
MINUTE_COLUMNS = [
    "camera_id", "start_utc", "cell_m", "minute", "person_s", "observed_s", "incomplete"
]  # fmt: skip


def run(path):
    store = SqliteSlotStore(path)
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0)
    for i in range(50):
        r.add(T0 + i, np.arange(6), np.linspace(0.1, 1.9, 6), np.full(6, 0.3), seconds=1.0)
    r.close()
    store.close()


def test_the_database_holds_only_the_slots_and_minute_totals_tables(tmp_path):
    run(tmp_path / "x.db")

    with sqlite3.connect(tmp_path / "x.db") as db:
        objects = db.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        columns = [row[1] for row in db.execute("PRAGMA table_info(slots)")]
        minute_columns = [row[1] for row in db.execute("PRAGMA table_info(slot_minutes)")]

    assert sorted(objects) == [("table", "slot_minutes"), ("table", "slots")]
    assert columns == COLUMNS
    assert minute_columns == MINUTE_COLUMNS


def test_minute_totals_are_one_number_per_minute_for_the_whole_grid(tmp_path):
    run(tmp_path / "x.db")

    with sqlite3.connect(tmp_path / "x.db") as db:
        rows = db.execute("SELECT minute, person_s, observed_s FROM slot_minutes").fetchall()

    assert sorted(r[0] for r in rows) == list(range(15))  # one row per minute of the slot
    assert all(isinstance(p, float) and isinstance(o, float) for _, p, o in rows)
    assert sum(o for _, _, o in rows) == 50.0  # 50 one-second frames


def test_a_grid_blob_is_one_floor_grid(tmp_path):
    run(tmp_path / "x.db")

    with sqlite3.connect(tmp_path / "x.db") as db:
        rows = db.execute("SELECT layer, grid_blob, n_unique FROM slots").fetchall()

    assert sorted(r[0] for r in rows) == ["dwell", "passage"]
    for _, blob, n_unique in rows:
        grid = SqliteSlotStore.decode(blob)
        assert grid.shape == (SPEC.ny, SPEC.nx) and grid.dtype == np.float64
        assert n_unique == 6


def test_start_utc_reads_as_an_iso_quarter(tmp_path):
    run(tmp_path / "x.db")

    with sqlite3.connect(tmp_path / "x.db") as db:
        starts = {r[0] for r in db.execute("SELECT start_utc FROM slots")}

    assert starts == {"2026-11-06T09:00:00Z"}


def test_a_second_run_over_the_same_slot_replaces_it(tmp_path):
    run(tmp_path / "x.db")
    run(tmp_path / "x.db")

    with sqlite3.connect(tmp_path / "x.db") as db:
        assert db.execute("SELECT count(*) FROM slots").fetchone() == (2,)


def slot(n_unique, incomplete):
    grid = np.full((SPEC.ny, SPEC.nx), float(n_unique))
    return [
        SlotGrid("cam", T0, 0.5, layer, grid, n_unique, incomplete)
        for layer in ("dwell", "passage")
    ]


def stored(store):
    return {(s.n_unique, s.incomplete) for s in store.read("cam", T0, T0 + 900)}


def test_an_incomplete_slot_never_replaces_a_complete_one():
    """A clock stepping back re-enters a quarter already written: the partial
    second pass must not overwrite the full first one."""
    store = SqliteSlotStore(":memory:")
    store.write(slot(9, incomplete=False))
    store.write(slot(5, incomplete=True))

    assert stored(store) == {(9, False)}


def test_a_complete_slot_replaces_an_incomplete_one():
    store = SqliteSlotStore(":memory:")
    store.write(slot(5, incomplete=True))
    store.write(slot(9, incomplete=False))

    assert stored(store) == {(9, False)}


def test_two_incomplete_passes_merge_without_overstating_presence():
    """A process restart in one quarter writes it twice, both incomplete. Ids
    differ across runs, so dwell adds up while passage and n_unique take the
    larger pass: no dwell is lost and no person is counted twice."""
    store = SqliteSlotStore(":memory:")
    store.write(slot(9, incomplete=True))  # first 10 min, then the process stops
    store.write(slot(5, incomplete=True))  # restart, last 3 min

    rows = {s.layer: s for s in store.read("cam", T0, T0 + 900)}
    assert stored(store) == {(9, True)}
    assert np.all(rows["dwell"].grid == 14.0)
    assert np.all(rows["passage"].grid == 9.0)
