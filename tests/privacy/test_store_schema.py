"""R4 (spec 3): only aggregated grids are persisted, never trajectories or images.

The database a run writes holds one table, slots, with exactly the contract's
columns; a grid blob decodes to one floor grid of numbers, nothing per track.
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


def run(path):
    store = SqliteSlotStore(path)
    r = RecordSlots(store, "cam", SPEC, retention_days=90, now=lambda: T0)
    for i in range(50):
        r.add(T0 + i, np.arange(6), np.linspace(0.1, 1.9, 6), np.full(6, 0.3), seconds=1.0)
    r.close()
    store.close()


def test_the_database_holds_only_the_slots_table(tmp_path):
    run(tmp_path / "x.db")

    with sqlite3.connect(tmp_path / "x.db") as db:
        objects = db.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        columns = [row[1] for row in db.execute("PRAGMA table_info(slots)")]

    assert objects == [("table", "slots")]
    assert columns == COLUMNS


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
