"""Use case: turn floor points of tracked people into persisted 15-min slots and the live layer.

A slot is written when time moves into the next quarter, and on close. It is
marked incomplete when the source did not cover it: its first sample came
more than one frame after the slot began, the source dropped while it was
open, or it was still open when the source ended. Each write also purges the
slots past the retention (R7); a slot already past it when it closes (a video
with an old start) is not written, only counted in past_retention.

With a stitch, k counts the slot's stitched chains instead of raw ids (phase 6,
H3): the stitch runs when the slot closes, over what the open slot kept of its
tracks in memory. The live layer keeps counting raw ids over its 2 minutes.
"""

import numpy as np

from space_usage_heatmaps.occupancy.application.purge_slots import (
    DAY_S,
    RETENTION_DAYS,
    purge_slots,
)
from space_usage_heatmaps.occupancy.domain.live import LiveLayer
from space_usage_heatmaps.occupancy.domain.slots import (
    K_MIN_PEOPLE,
    SlotAccumulator,
    follow_rows,
    slot_start,
)


class RecordSlots:
    def __init__(
        self,
        store,
        camera_id,
        spec,
        retention_days=RETENTION_DAYS,
        now=None,
        k=K_MIN_PEOPLE,
        stitch=None,
    ):
        """now: the wall clock (epoch seconds) the retention counts from;
        stitch: k counts chains of this stitch (None: raw ids)."""
        self.store, self.camera_id, self.spec, self.k = store, camera_id, spec, k
        self.stitch = stitch
        self.retention_days, self.now = retention_days, now
        self.slot = None
        self.written = 0  # slots written so far
        self.past_retention = 0  # slots not written: older than the retention
        self.live_layer = LiveLayer((spec.ny, spec.nx), k=k)

    def add(self, t, track_ids, xs, ys, seconds):
        """Samples of one frame at t (UTC epoch seconds), each worth seconds of dwell;
        call it for frames without anyone too, so slots close on time."""
        start = slot_start(t)
        if self.slot is not None and start != self.slot.start_utc:
            self._write()
        if self.slot is None:
            self.slot = SlotAccumulator(
                self.spec, start, incomplete=t - start > seconds, stitch=self.stitch
            )
        follow_rows(self.slot, t, track_ids, xs, ys, seconds)
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
        """(dwell, passage) of the live layer at t, decayed with a 60 s half-life,
        empty where fewer than k people were seen in the last 120 s (R5)."""
        return self.live_layer.at(t)

    def _write(self):
        slot, self.slot = self.slot, None
        now = None if self.now is None else self.now()
        if now is not None and slot.start_utc < now - self.retention_days * DAY_S:
            self.past_retention += 1
            return
        self.store.write(slot.grids(self.camera_id, self.k))
        self.written += 1
        if now is not None:
            purge_slots(self.store, now, self.retention_days)
