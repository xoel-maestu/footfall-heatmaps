"""Use case: turn floor points of tracked people into persisted 15-min slots and the live layer.

A slot is written when time moves into the next quarter, and on close. It is
marked incomplete when the source did not cover it: its first sample came
more than one frame after the slot began, the source dropped while it was
open, or it was still open when the source ended. Each write also purges the
slots past the retention (R7).
"""

import numpy as np

from space_usage_heatmaps.occupancy.application.purge_slots import RETENTION_DAYS, purge_slots
from space_usage_heatmaps.occupancy.domain.live import LiveLayer
from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, SlotAccumulator, slot_start


class RecordSlots:
    def __init__(
        self, store, camera_id, spec, retention_days=RETENTION_DAYS, now=None, k=K_MIN_PEOPLE
    ):
        """now: the wall clock (epoch seconds) the retention counts from."""
        self.store, self.camera_id, self.spec, self.k = store, camera_id, spec, k
        self.retention_days, self.now = retention_days, now
        self.slot = None
        self.live_layer = LiveLayer((spec.ny, spec.nx))

    def add(self, t, track_ids, xs, ys, seconds):
        """Samples of one frame at t (UTC epoch seconds), each worth seconds of dwell;
        call it for frames without anyone too, so slots close on time."""
        start = slot_start(t)
        if self.slot is not None and start != self.slot.start_utc:
            self._write()
        if self.slot is None:
            self.slot = SlotAccumulator(self.spec, start, incomplete=t - start > seconds)
        cells, inside = self.spec.cell_index(xs, ys)
        ids = np.asarray(track_ids)[inside]
        self.slot.add(ids, cells, seconds)
        self.live_layer.add(t, ids, cells, seconds)

    def mark_incomplete(self):
        """The source dropped: the open slot misses part of its time."""
        if self.slot is not None:
            self.slot.incomplete = True

    def close(self):
        """The source ended: the open slot is written as incomplete."""
        if self.slot is not None:
            self.slot.incomplete = True
            self._write()

    def live(self, t):
        """(dwell, passage) of the live layer at t, decayed with a 60 s half-life."""
        return self.live_layer.at(t)

    def _write(self):
        self.store.write(self.slot.grids(self.camera_id, self.k))
        self.slot = None
        if self.now is not None:
            purge_slots(self.store, self.now(), self.retention_days)
