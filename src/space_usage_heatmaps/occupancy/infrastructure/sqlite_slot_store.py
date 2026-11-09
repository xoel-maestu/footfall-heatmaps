"""SlotStore over SQLite (standard library): aggregated grids only (R4).

slots holds one floor grid per slot and layer; slot_minutes one row per minute
of a slot, two numbers for the whole grid (person-seconds over every cell, 0 below k ids,
and observed seconds), nothing per cell or per track.
"""

import io
import sqlite3
from datetime import UTC, datetime

import numpy as np

from space_usage_heatmaps.occupancy.domain.slots import (
    MINUTE_S,
    MINUTES,
    SlotGrid,
    SlotMinutes,
    merge_incomplete,
    merge_incomplete_minutes,
)

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
MINUTES_SCHEMA = """CREATE TABLE IF NOT EXISTS slot_minutes (
    camera_id TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    cell_m REAL NOT NULL,
    minute INTEGER NOT NULL CHECK (minute >= 0 AND minute < 15),
    person_s REAL NOT NULL,
    observed_s REAL NOT NULL,
    incomplete INTEGER NOT NULL CHECK (incomplete IN (0, 1)),
    PRIMARY KEY (camera_id, start_utc, cell_m, minute)
)"""


def _iso(t):
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(iso):
    return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp())


class SqliteSlotStore:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute(SCHEMA)
        self.db.execute(MINUTES_SCHEMA)
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

    def write(self, slots, minutes=()):
        with self.db:
            for m in minutes:
                self._write_minutes(m)
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

    def _write_minutes(self, m):
        key = (m.camera_id, _iso(m.start_utc), m.cell_m)
        if m.incomplete:
            old = self._minutes_rows(*key)
            if old is not None and not old.incomplete:
                return  # a complete slot's minutes are never replaced by an incomplete pass
            if old is not None:
                m = merge_incomplete_minutes(old, m)
        # Overlapping incomplete passes can add up to more than a minute: clamp the
        # observed time to 60 s and scale the dwell in the same ratio, so the
        # minute's occupancy (person_s / observed_s) is what the passes read.
        raw = np.asarray(m.observed_s, float)
        observed = np.minimum(raw, MINUTE_S)
        scale = np.divide(observed, raw, out=np.ones(MINUTES), where=raw > MINUTE_S)
        person_s = np.asarray(m.person_s, float) * scale
        self.db.executemany(
            "INSERT OR REPLACE INTO slot_minutes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (*key, i, float(person_s[i]), float(observed[i]), int(m.incomplete))
                for i in range(MINUTES)
            ],
        )

    def _minutes_rows(self, camera_id, start_iso, cell_m):
        rows = self.db.execute(
            "SELECT minute, person_s, observed_s, incomplete FROM slot_minutes"
            " WHERE camera_id = ? AND start_utc = ? AND cell_m = ?",
            (camera_id, start_iso, cell_m),
        ).fetchall()
        return _minutes(camera_id, start_iso, cell_m, rows) if rows else None

    def read_minutes(self, camera_id, start_utc, end_utc):
        keys = self.db.execute(
            "SELECT DISTINCT start_utc, cell_m FROM slot_minutes WHERE camera_id = ?"
            " AND start_utc >= ? AND start_utc < ? ORDER BY start_utc, cell_m",
            (camera_id, _iso(start_utc), _iso(end_utc)),
        ).fetchall()
        return [self._minutes_rows(camera_id, s, m) for s, m in keys]

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
            self.db.execute("DELETE FROM slot_minutes WHERE start_utc < ?", (_iso(before_utc),))
            return self.db.execute(
                "DELETE FROM slots WHERE start_utc < ?", (_iso(before_utc),)
            ).rowcount

    def close(self):
        self.db.close()


def _minutes(camera_id, start_iso, cell_m, rows):
    person_s, observed_s = np.zeros(MINUTES), np.zeros(MINUTES)
    for minute, p, o, _ in rows:
        person_s[minute], observed_s[minute] = p, o
    incomplete = any(i for *_, i in rows)
    return SlotMinutes(camera_id, _epoch(start_iso), cell_m, person_s, observed_s, incomplete)
