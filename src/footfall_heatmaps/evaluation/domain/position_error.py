"""Ground position error of per-sample floor points against the floor truth."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PositionError:
    median: float | None  # metres; None when no sample matches the truth
    p90: float | None
    matched: int  # samples with a truth row of the same track and frame
    unmatched: int

    def as_dict(self):
        return asdict(self)


def position_error(ids, frames, xy, truth_ids, truth_frames, truth_xy):
    """Distance of each (track id, frame) sample to the truth row with the same
    track id and frame; samples without one are counted, not scored."""
    span = int(max(frames.max(initial=0), truth_frames.max(initial=0))) + 1
    truth_keys = truth_ids.astype(np.int64) * span + truth_frames
    order = np.argsort(truth_keys)
    sorted_keys = truth_keys[order]
    keys = ids.astype(np.int64) * span + frames
    at = np.clip(np.searchsorted(sorted_keys, keys), 0, max(len(sorted_keys) - 1, 0))
    found = sorted_keys[at] == keys if len(sorted_keys) else np.zeros(len(keys), dtype=bool)
    d = np.hypot(*(np.asarray(xy)[found] - truth_xy[order[at[found]]]).T)
    if not len(d):
        return PositionError(None, None, 0, int(len(keys)))
    return PositionError(
        float(np.median(d)), float(np.percentile(d, 90)), int(len(d)), int((~found).sum())
    )
