"""Person detections of a detector recipe saved under the data dir, so tracker
variants rerun in minutes instead of running the detector again. Boxes and
scores per processed frame only, never an image (privacy rules R1, R3), plus the
recipe metadata (weights, input size, thresholds, runtime) and the video's
identity, checked on load as in track_cache: a file whose metadata differs, or
that cannot be read, is recomputed."""

import json
import sys

import numpy as np

from footfall_heatmaps.evaluation.infrastructure.track_cache import (
    mislabelled,
    read_cache,
    stale,
)
from footfall_heatmaps.perception.domain.frame import Detections


def _stderr(line):
    print(line, file=sys.stderr)


class CachedDetections:
    def __init__(self, source, directory, metadata, log=_stderr, identity=None):
        """source.detections(scene, camera) -> (frame index, Detections) of every
        processed frame; called only on a miss. source.last_run, if any, is how
        that run went (the providers the detector ran on), stored with the file.
        identity(scene, camera): what identifies that camera's video (its size),
        checked with the recipe."""
        self._source = source
        self.directory = directory
        self.metadata = dict(metadata)
        self._log = log
        self._identity = identity or (lambda scene, camera: {})
        self.last_run = None

    def has(self, scene, camera):
        """Whether a stored file of this recipe holds the camera's detections."""
        path = self.directory / f"scene_{scene}_camera_{camera}.npz"
        if not path.exists():
            return False
        read = read_cache(path, ("frames",))
        expected = self.metadata | self._identity(scene, camera)
        return read is not None and stale(read[0], expected, lambda line: None, "") is None

    def detections(self, scene, camera):
        path = self.directory / f"scene_{scene}_camera_{camera}.npz"
        expected = self.metadata | self._identity(scene, camera)
        if path.exists():
            read = read_cache(path, ("frames", "box_frames", "boxes", "scores"))
            why = "unreadable" if read is None else None
            if read is not None:
                stored, f = read
                if (why := stale(stored, expected, self._log, "cached detections")) is None:
                    self.last_run = stored.get("run")
                    return _replay(f["frames"], f["box_frames"], f["boxes"], f["scores"])
            self._log(f"cached detections {path.name}: {why}; recomputing")
        run = list(self._source.detections(scene, camera))
        self.last_run = getattr(self._source, "last_run", None)
        frames = np.array([i for i, _ in run], dtype=int)
        box_frames = np.concatenate([np.full(len(d.scores), i) for i, d in run] + [[]]).astype(int)
        boxes = np.concatenate([d.boxes.reshape(-1, 4) for _, d in run] + [np.empty((0, 4))])
        scores = np.concatenate([d.scores for _, d in run] + [[]]).astype(float)
        if (why := mislabelled(expected, self.last_run)) is not None:
            self._log(f"cached detections {path.name}: {why}; not stored")
            return run
        stored = expected | ({"run": self.last_run} if self.last_run else {})
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        meta = np.array(json.dumps(stored, sort_keys=True))
        np.savez(tmp, frames=frames, box_frames=box_frames, boxes=boxes, scores=scores, meta=meta)
        tmp.replace(path)
        return run


def _replay(frames, box_frames, boxes, scores):
    """The (frame index, Detections) list again; box_frames is sorted."""
    lo = np.searchsorted(box_frames, frames, side="left")
    hi = np.searchsorted(box_frames, frames, side="right")
    return [
        (int(i), Detections(boxes[a:b], scores[a:b]))
        for i, a, b in zip(frames, lo, hi, strict=True)
    ]


class StridedDetections:
    """A stride-N recipe's detections: the every-frame cache when it holds the
    camera (a stride subsamples it exactly), else the detector on every N-th
    frame only, in its own cache."""

    def __init__(self, every_frame, strided):
        self._every, self._strided = every_frame, strided
        self.last_run = None

    def detections(self, scene, camera):
        source = self._every if self._every.has(scene, camera) else self._strided
        run = source.detections(scene, camera)
        self.last_run = source.last_run
        return run
