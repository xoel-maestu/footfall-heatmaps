"""Tracks of a configuration saved under the data dir, so re-scoring does not
run the detector again. Positions only (track id, frame and box per row), plus
the metadata of the recipe that produced them (library version, tracker
settings, classes, confidence, input size, weights), checked on load: a file
whose metadata differs from the current recipe, or that cannot be read, is
recomputed."""

import json
import sys
import zipfile

import numpy as np

from space_usage_heatmaps.evaluation.domain.tracks import Tracks


def _stderr(line):
    print(line, file=sys.stderr)


class CachedTrackSource:
    def __init__(self, make_source, directory, metadata=None, log=_stderr, identity=None):
        """make_source() builds the real source; called only on a cache miss.
        metadata: what the tracks depend on; a None value is unknown here (the
        library is not installed) and is not checked. identity(scene, camera):
        what one file depends on besides the recipe (the video, the camera's
        calibration), checked the same way."""
        self._make = make_source
        self._source = None
        self.directory = directory
        self.metadata = dict(metadata or {})
        self._log = log
        self._identity = identity or (lambda scene, camera: {})
        self.provenance = {}  # "scene/camera" -> stored metadata and unchecked keys

    def tracks(self, scene, camera):
        path = self.directory / f"scene_{scene}_camera_{camera}.npz"
        expected = self.metadata | self._identity(scene, camera)
        if path.exists():
            read = read_cache(path, ("track_ids", "frames", "boxes"))
            why = "unreadable" if read is None else None
            if read is not None:
                stored, f = read
                if (why := stale(stored, expected, self._log, "cached tracks")) is None:
                    unknown = sorted(k for k, v in expected.items() if v is None)
                    self.provenance[f"{scene}/{camera}"] = {"stored": stored, "unverified": unknown}
                    return Tracks(f["track_ids"], f["frames"], f["boxes"])
            self._log(f"cached tracks {path.name}: {why}; recomputing")
        if self._source is None:
            self._source = self._make()
        t = self._source.tracks(scene, camera)
        # How this computation went (the source's last_run, e.g. the provider the
        # detector ran on): stored and reported, not part of the recipe check.
        run = getattr(self._source, "last_run", None)
        if (why := mislabelled(expected, run)) is not None:
            self._log(f"cached tracks {path.name}: {why}; not stored")
            return t
        stored = expected | ({"run": run} if run else {})
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        meta = np.array(json.dumps(stored, sort_keys=True))
        np.savez(tmp, track_ids=t.track_ids, frames=t.frames, boxes=t.boxes, meta=meta)
        tmp.replace(path)
        self.provenance[f"{scene}/{camera}"] = {"stored": stored, "unverified": []}
        return t


def mislabelled(metadata, run):
    """Why a result computed on run's providers must not be stored under
    metadata, or None. An onnx-trt recipe is stored only if every frame ran on
    TensorRT: a fallback to CUDA would replay as TensorRT under its key."""
    providers = (run or {}).get("providers")
    if metadata.get("engine") == "onnx-trt" and providers != ["TensorrtExecutionProvider"]:
        return f"ran on {' then '.join(providers or ['unknown providers'])}, not TensorRT"
    return None


def read_cache(path, keys):
    """(stored metadata or None, {key: array}) of a cache file, or None when it
    cannot be read or lacks one of keys (a truncated or foreign file)."""
    try:
        with np.load(path, allow_pickle=False) as f:
            stored = json.loads(str(f["meta"])) if "meta" in f.files else None
            return stored, {k: f[k] for k in keys}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None


def stale(stored, metadata, log, what):
    """Why the stored metadata does not match the expected one, or None when it
    does. An expected None is unknown here (not installed): logged, not checked."""
    if stored is None:
        return "no metadata"
    diffs = [
        f"{k} {stored.get(k)!r} != {v!r}"
        for k, v in metadata.items()
        if v is not None and stored.get(k) != v
    ]
    if diffs:
        return ", ".join(diffs)
    if unknown := [k for k, v in metadata.items() if v is None]:
        log(f"{what}: {', '.join(unknown)} not checked (not installed here)")
    return None
