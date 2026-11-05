"""Tracks of a configuration saved under the data dir, so re-scoring does not
run the detector again. Positions only (track id, frame and box per row), plus
the metadata of the recipe that produced them (library version, tracker
settings, classes, confidence, input size, weights), checked on load: a file
whose metadata differs from the current recipe is recomputed."""

import json
import sys

import numpy as np

from space_usage_heatmaps.evaluation.domain.tracks import Tracks


def _stderr(line):
    print(line, file=sys.stderr)


class CachedTrackSource:
    def __init__(self, make_source, directory, metadata=None, log=_stderr):
        """make_source() builds the real source; called only on a cache miss.
        metadata: what the tracks depend on; a None value is unknown here (the
        library is not installed) and is not checked."""
        self._make = make_source
        self._source = None
        self.directory = directory
        self.metadata = dict(metadata or {})
        self._log = log
        self.provenance = {}  # "scene/camera" -> stored metadata and unchecked keys

    def tracks(self, scene, camera):
        path = self.directory / f"scene_{scene}_camera_{camera}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                stored = json.loads(str(f["meta"])) if "meta" in f.files else None
                if (why := self._stale(stored)) is None:
                    unknown = sorted(k for k, v in self.metadata.items() if v is None)
                    self.provenance[f"{scene}/{camera}"] = {"stored": stored, "unverified": unknown}
                    return Tracks(f["track_ids"], f["frames"], f["boxes"])
            self._log(f"cached tracks {path.name}: {why}; recomputing")
        if self._source is None:
            self._source = self._make()
        t = self._source.tracks(scene, camera)
        # How this computation went (the source's last_run, e.g. the provider the
        # detector ran on): stored and reported, not part of the recipe check.
        run = getattr(self._source, "last_run", None)
        stored = self.metadata | ({"run": run} if run else {})
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        meta = np.array(json.dumps(stored, sort_keys=True))
        np.savez(tmp, track_ids=t.track_ids, frames=t.frames, boxes=t.boxes, meta=meta)
        tmp.replace(path)
        self.provenance[f"{scene}/{camera}"] = {"stored": stored, "unverified": []}
        return t

    def _stale(self, stored):
        """Why the stored metadata does not match, or None when it does."""
        if stored is None:
            return "no metadata"
        diffs = [
            f"{k} {stored.get(k)!r} != {v!r}"
            for k, v in self.metadata.items()
            if v is not None and stored.get(k) != v
        ]
        if diffs:
            return ", ".join(diffs)
        if unknown := [k for k, v in self.metadata.items() if v is None]:
            self._log(f"cached tracks: {', '.join(unknown)} not checked (not installed here)")
        return None
