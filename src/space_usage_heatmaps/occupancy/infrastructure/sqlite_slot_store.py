"""SlotStore over SQLite (standard library): one table, aggregated grids only (R4)."""

import io
import sqlite3
from datetime import UTC, datetime

import numpy as np

from space_usage_heatmaps.occupancy.domain.slots import SlotGrid

SCHEMA = """CREATE TABLE IF NOT EXISTS slots (
    camera_id TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    cell_m REAL NOT NULL,
    layer TEXT NOT NULL CHECK (layer IN ('dwell', 'passage')),
    grid_blob BLOB NOT NULL,
    n_unique INTEGER NOT NULL,
    incomplete INTEGER NOT NULL CHECK (incomplete IN (0, 1)),
    PRIMARY KEY (camera_id, start_utc, cell_m, layer)
)"""


def _iso(t):
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(iso):
    return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp())


class SqliteSlotStore:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute(SCHEMA)
        self.db.commit()

    @staticmethod
    def encode(grid):
        """A .npy of the float64 grid: its shape and type travel with it."""
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(grid, np.float64), allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def decode(blob):
        return np.load(io.BytesIO(blob), allow_pickle=False)

    def write(self, slots):
        with self.db:
            self.db.executemany(
                # A complete slot is never replaced by an incomplete one, e.g. a
                # stream whose wall clock steps back into a quarter already written.
                "INSERT INTO slots VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (camera_id, start_utc, cell_m, layer) DO UPDATE SET"
                " grid_blob = excluded.grid_blob, n_unique = excluded.n_unique,"
                " incomplete = excluded.incomplete"
                " WHERE slots.incomplete = 1 OR excluded.incomplete = 0",
                [
                    (
                        s.camera_id,
                        _iso(s.start_utc),
                        s.cell_m,
                        s.layer,
                        self.encode(s.grid),
                        s.n_unique,
                        int(s.incomplete),
                    )
                    for s in slots
                ],
            )

    def read(self, camera_id, start_utc, end_utc):
        rows = self.db.execute(
            "SELECT camera_id, start_utc, cell_m, layer, grid_blob, n_unique, incomplete"
            " FROM slots WHERE camera_id = ? AND start_utc >= ? AND start_utc < ?"
            " ORDER BY start_utc, cell_m, layer",
            (camera_id, _iso(start_utc), _iso(end_utc)),
        )
        return [
            SlotGrid(c, _epoch(s), m, layer, self.decode(b), n, bool(i))
            for c, s, m, layer, b, n, i in rows
        ]

    def purge(self, before_utc):
        with self.db:
            return self.db.execute(
                "DELETE FROM slots WHERE start_utc < ?", (_iso(before_utc),)
            ).rowcount

    def close(self):
        self.db.close()
