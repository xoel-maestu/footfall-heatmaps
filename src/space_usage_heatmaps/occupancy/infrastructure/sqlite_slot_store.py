"""SlotStore over SQLite (standard library): one table, aggregated grids only (R4)."""

import io
import sqlite3
from datetime import UTC, datetime

import numpy as np

from space_usage_heatmaps.occupancy.domain.slots import SlotGrid, merge_incomplete

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
            for s in slots:
                if s.incomplete:
                    row = self.db.execute(
                        "SELECT grid_blob, n_unique, incomplete FROM slots WHERE camera_id = ?"
                        " AND start_utc = ? AND cell_m = ? AND layer = ?",
                        (s.camera_id, _iso(s.start_utc), s.cell_m, s.layer),
                    ).fetchone()
                    if row is not None and not row[2]:
                        continue  # a complete slot is never replaced by an incomplete one
                    if row is not None:
                        old = SlotGrid(
                            s.camera_id,
                            s.start_utc,
                            s.cell_m,
                            s.layer,
                            self.decode(row[0]),
                            row[1],
                            True,
                        )
                        s = merge_incomplete(old, s)
                self.db.execute(
                    "INSERT OR REPLACE INTO slots VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        s.camera_id,
                        _iso(s.start_utc),
                        s.cell_m,
                        s.layer,
                        self.encode(s.grid),
                        s.n_unique,
                        int(s.incomplete),
                    ),
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

    def latest_start(self, camera_id):
        """Epoch seconds of the newest slot start of the camera, or None."""
        (start,) = self.db.execute(
            "SELECT MAX(start_utc) FROM slots WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        return None if start is None else _epoch(start)

    def purge(self, before_utc):
        with self.db:
            return self.db.execute(
                "DELETE FROM slots WHERE start_utc < ?", (_iso(before_utc),)
            ).rowcount

    def close(self):
        self.db.close()
