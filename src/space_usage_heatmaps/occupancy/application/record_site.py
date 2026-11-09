"""Use case: the cameras of one site into fused 15-min slots and a fused live layer.

Each camera keeps its own open slot (its ephemeral ids, in memory only) and
its own live layer. A quarter is closed when every camera has moved past it,
or when the newest camera is more than one slot ahead of it (a camera that
stays down does not hold the others' slots open), or when the sources end.
On close the cameras' raw grids are fused with the site's frozen weights
(occupancy.domain.fusion, from a calibration window), and k=5 applies to the
fused grid: a cell shows only where the fused passage, the weighted mean of
the covering cameras' distinct people, reaches k. Only that is written, under
the site id; nothing per camera or per track is stored.

A fused slot is incomplete when a camera joined it late, dropped while it was
open, never reported it, or when the sources ended inside it. Samples a
camera sends for a quarter already written are dropped (counted in late).
n_unique is the most distinct ids any one camera saw: ids are per camera, so
adding them would count a person seen by two cameras twice.

The slot's minute totals (phase 6, H4) are the cameras' per-minute dwell
fused with the same weights, summed over the whole grid (hidden cells too: a
floor total places no one); a minute reads as 0 when no camera saw k distinct
ids in it. In each minute only the cameras that covered it (observed time > 0)
are fused, their weights renormalised per cell over those cameras, so a camera
that is down for a minute does not dilute it with zeros.
Its observed seconds are the most any one camera covered in that minute.
"""

from dataclasses import replace

import numpy as np

from space_usage_heatmaps.occupancy.application.purge_slots import (
    DAY_S,
    RETENTION_DAYS,
    purge_slots,
)
from space_usage_heatmaps.occupancy.domain.fusion import fuse_layers
from space_usage_heatmaps.occupancy.domain.live import LiveLayer
from space_usage_heatmaps.occupancy.domain.slots import (
    K_MIN_PEOPLE,
    LAYERS,
    MINUTES,
    SLOT_S,
    SlotAccumulator,
    SlotGrid,
    SlotMinutes,
    slot_start,
    suppress,
    suppress_count,
)


class RecordSite:
    def __init__(
        self,
        store,
        site_id,
        cameras,
        weights,
        spec,
        retention_days=RETENTION_DAYS,
        now=None,
        k=K_MIN_PEOPLE,
    ):
        """weights: (cameras, ny, nx), the site's frozen fusion weights in the
        order of cameras; now: the wall clock the retention counts from."""
        self.cameras = [str(c) for c in cameras]
        self.weights = np.asarray(weights, float)
        if self.weights.shape != (len(self.cameras), spec.ny, spec.nx):
            raise ValueError(
                f"fusion weights of shape {self.weights.shape}, but the site has "
                f"{len(self.cameras)} cameras on a {spec.ny}x{spec.nx} grid"
            )
        self.store, self.site_id, self.spec, self.k = store, site_id, spec, k
        self.retention_days, self.now = retention_days, now
        self._index = {c: i for i, c in enumerate(self.cameras)}
        self._open = {}  # slot start -> {camera: SlotAccumulator}
        self._current = {}  # camera -> the slot start of its newest sample
        self._closed_until = None  # every quarter before this was written
        self.written = 0
        self.past_retention = 0
        self.late = 0  # samples dropped: their quarter was already written
        self._live = {c: LiveLayer((spec.ny, spec.nx), k=k) for c in self.cameras}

    def add(self, camera_id, t, track_ids, xs, ys, seconds):
        """One frame of one camera at t (UTC epoch seconds); call it for frames
        without anyone too, so slots close on time. KeyError: not a site camera."""
        if camera_id not in self._index:
            raise KeyError(f"camera {camera_id!r} is not in site {self.site_id}")
        start = slot_start(t)
        cells, inside = self.spec.cell_index(xs, ys)
        ids = np.asarray(track_ids)[inside]
        self._live[camera_id].add(t, ids, cells, seconds)
        if self._closed_until is not None and start < self._closed_until:
            self.late += len(ids)
            return
        slot = self._open.setdefault(start, {})
        if camera_id not in slot:
            slot[camera_id] = SlotAccumulator(self.spec, start, incomplete=t - start > seconds)
        slot[camera_id].add(ids, cells, seconds, t)
        self._current[camera_id] = max(start, self._current.get(camera_id, start))
        self._close_passed()

    def mark_incomplete(self, camera_id):
        """The camera's source dropped: its open slot misses part of its time."""
        start = self._current.get(camera_id)
        if start is not None and camera_id in self._open.get(start, {}):
            self._open[start][camera_id].incomplete = True

    def close(self):
        """The sources ended: every open slot is written as incomplete."""
        for start in sorted(self._open):
            self._write(start, ended=True)

    def open_starts(self):
        return sorted(self._open)

    def live(self, t):
        """(dwell, passage) of the fused live layer at t: each camera's layer
        decayed (60 s half-life), fused with the site weights, then empty where
        the fused count of people seen in the last 120 s is under k (R5)."""
        raw = [self._live[c].raw(t) for c in self.cameras]
        dwell, passage, people = (fuse_layers(self.weights, [r[i] for r in raw]) for i in range(3))
        dwell, _ = suppress(dwell, people, self.k)
        passage, _ = suppress(passage, people, self.k)
        return dwell, passage

    def _close_passed(self):
        every = min(self._current.values()) if len(self._current) == len(self.cameras) else None
        newest = max(self._current.values())
        for start in sorted(self._open):
            if (every is not None and start < every) or start + SLOT_S < newest:
                self._write(start)

    def _minutes(self, slot, start):
        """The fused slot's minute totals over the whole grid, 0 below k ids."""
        cells = self.spec.ny * self.spec.nx
        shown = np.ones(cells, bool)
        weights = self.weights.reshape(len(self.cameras), cells)
        dwell = np.zeros((len(self.cameras), MINUTES, cells))
        covered = np.zeros((len(self.cameras), MINUTES), bool)
        for i, c in enumerate(self.cameras):
            if c in slot:
                dwell[i], covered[i] = slot[c].minute_dwell, slot[c].observed > 0
        # (cameras, minutes, cells): each camera's weight where it covered the minute,
        # renormalised per cell over those cameras
        w = weights[:, None, :] * covered[:, :, None]
        total = w.sum(axis=0)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        fused = (w * dwell).sum(axis=0)
        ids = np.zeros(MINUTES, int)
        observed = np.zeros(MINUTES)
        for a in slot.values():
            ids = np.maximum(ids, a.minute_raw(shown)[1])
            observed = np.maximum(observed, a.observed)
        person_s = np.where(ids < self.k, 0.0, fused[:, shown].sum(axis=1))
        return SlotMinutes(self.site_id, start, self.spec.cell_m, person_s, observed, False)

    def _write(self, start, ended=False):
        slot = self._open.pop(start)
        self._closed_until = max(start + SLOT_S, self._closed_until or 0)
        now = None if self.now is None else self.now()
        if now is not None and start < now - self.retention_days * DAY_S:
            self.past_retention += 1
            return
        shape = (self.spec.ny, self.spec.nx)
        raws = [slot[c].raw() if c in slot else (np.zeros(shape), np.zeros(shape))
                for c in self.cameras]  # fmt: skip
        dwell = fuse_layers(self.weights, [r[0] for r in raws])
        passage = fuse_layers(self.weights, [r[1] for r in raws])
        minutes = self._minutes(slot, start)
        dwell, passage = suppress(dwell, passage, self.k)
        n = suppress_count(max(len(a.ids) for a in slot.values()), self.k)
        incomplete = (
            ended or len(slot) < len(self.cameras) or any(a.incomplete for a in slot.values())
        )
        self.store.write(
            [
                SlotGrid(self.site_id, start, self.spec.cell_m, layer, grid, n, incomplete)
                for layer, grid in zip(LAYERS, (dwell, passage), strict=True)
            ],
            [replace(minutes, incomplete=incomplete)],
        )
        self.written += 1
        if now is not None:
            purge_slots(self.store, now, self.retention_days)
