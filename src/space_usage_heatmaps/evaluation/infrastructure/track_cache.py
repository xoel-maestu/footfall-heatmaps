"""Tracks of a configuration saved under the data dir, so re-scoring does not
run the detector again. Positions only: track id, frame and box per row."""

import numpy as np

from space_usage_heatmaps.evaluation.domain.tracks import Tracks


class CachedTrackSource:
    def __init__(self, make_source, directory):
        """make_source() builds the real source; called only on a cache miss."""
        self._make = make_source
        self._source = None
        self.directory = directory

    def tracks(self, scene, camera):
        path = self.directory / f"scene_{scene}_camera_{camera}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                return Tracks(f["track_ids"], f["frames"], f["boxes"])
        if self._source is None:
            self._source = self._make()
        t = self._source.tracks(scene, camera)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, track_ids=t.track_ids, frames=t.frames, boxes=t.boxes)
        tmp.replace(path)
        return t
