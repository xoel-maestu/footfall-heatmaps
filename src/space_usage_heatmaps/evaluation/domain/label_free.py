"""Label-free metrics of spec 6: computed from the tracks alone, so they run on
any camera, and validated against the error the truth gives on NVIDIA.

- Occupancy conservation: people enter and leave the view at the edge of what
  the camera covers, the image border. The running balance of edge entries
  minus edge exits should equal the tracks active at each frame. A track born
  or lost inside the view (a fragment behind a shelf, a missed detection)
  breaks it. Error = sum over frames of |balance - active| / sum of active.
  Tracks present in the first frame count as entries and tracks still there in
  the last frame do not exit: the view does not start or end empty.
- Interior births: the share of tracks born after the first frame whose first
  box is away from the image border.
- Mean track length in seconds, first to last frame of each track.

A box is at the border when one of its edges is within EDGE_FRACTION of the
image's shorter side from the image edge. The fraction is fixed here, not
fitted to any dataset. Camera drift (edge similarity against a background
reference) needs the frames, which the evaluation does not keep: not computed.
"""

import math
from dataclasses import dataclass

import numpy as np

EDGE_FRACTION = 0.02  # of the shorter image side: 21.6 px at 1080p


@dataclass(frozen=True)
class LabelFree:
    conservation_error: float
    interior_birth_fraction: float  # NaN when no track is born after the first frame
    mean_track_length_s: float
    tracks: int

    def as_dict(self):
        return {k: (None if math.isnan(v) else v) for k, v in vars(self).items()}


@dataclass(frozen=True)
class LabelFreeBlocks:
    """Per block of consecutive frames; NaN where a block has nothing to measure."""

    conservation_error: np.ndarray
    interior_birth_fraction: np.ndarray  # of the tracks born in the block
    mean_track_length_s: np.ndarray  # of the tracks born in the block


def _at_edge(boxes, width, height):
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    e = EDGE_FRACTION * min(width, height)
    return (b[:, 0] <= e) | (b[:, 1] <= e) | (b[:, 2] >= width - e) | (b[:, 3] >= height - e)


def _per_frame(track_ids, frames, boxes, width, height, first, last):
    """(per-track birth, death, interior birth, |balance - active|, active per frame)."""
    ids, f = np.asarray(track_ids), np.asarray(frames)
    first = int(f.min()) if first is None else first
    last = int(f.max()) if last is None else last
    edge = _at_edge(boxes, width, height)
    order = np.lexsort((f, ids))
    ids, f, edge = ids[order], f[order], edge[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)] - 1
    birth, death = f[starts], f[ends]
    enters = edge[starts] | (birth <= first)
    leaves = edge[ends] & (death < last)
    n = last - first + 1
    entries = np.bincount(birth[enters] - first, minlength=n)[:n].cumsum()
    # An exit counts from the frame after the track's last one.
    exits = np.bincount(death[leaves] - first + 1, minlength=n + 1)[:n].cumsum()
    change = np.bincount(birth - first, minlength=n + 1) - np.bincount(
        death - first + 1, minlength=n + 1
    )
    active = change[:n].cumsum()
    interior = ~edge[starts] & (birth > first)
    return birth - first, death - first, interior, np.abs(entries - exits - active), active


def _mean(a):
    return float(np.mean(a)) if len(a) else math.nan


def label_free_metrics(track_ids, frames, boxes, width, height, fps, first=None, last=None):
    """The three metrics over one camera's tracks, frames first..last (default:
    the tracks' own range)."""
    birth, death, interior, gap, active = _per_frame(
        track_ids, frames, boxes, width, height, first, last
    )
    later = birth > 0
    return LabelFree(
        conservation_error=float(gap.sum() / active.sum()) if active.sum() else math.nan,
        interior_birth_fraction=_mean(interior[later]),
        mean_track_length_s=_mean((death - birth + 1) / fps),
        tracks=len(birth),
    )


def label_free_blocks(
    track_ids, frames, boxes, width, height, fps, block_frames, n_blocks, first=None, last=None
):
    """The same metrics per block of block_frames frames counted from frame 0
    (the blocks of the bootstrap): conservation over the block's frames, the
    other two over the tracks born in it."""
    first_frame = int(np.min(frames)) if first is None else first
    birth, death, interior, gap, active = _per_frame(
        track_ids, frames, boxes, width, height, first, last
    )
    frame_block = (np.arange(len(gap)) + first_frame) // block_frames
    birth_block = (birth + first_frame) // block_frames
    conservation = np.full(n_blocks, math.nan)
    interior_share = np.full(n_blocks, math.nan)
    length = np.full(n_blocks, math.nan)
    for k in range(n_blocks):
        here = frame_block == k
        if active[here].sum():
            conservation[k] = gap[here].sum() / active[here].sum()
        born = birth_block == k
        interior_share[k] = _mean(interior[born & (birth > 0)])
        length[k] = _mean((death[born] - birth[born] + 1) / fps)
    return LabelFreeBlocks(conservation, interior_share, length)


def _finite_pairs(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def pearson(x, y):
    """Pearson's r over the pairs where both values are finite; NaN without variance."""
    x, y = _finite_pairs(x, y)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(a):
    """Average ranks, 1-based; ties share the mean of their ranks."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    _, inverse = np.unique(a, return_inverse=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / np.bincount(inverse))[inverse]


def spearman(x, y):
    """Spearman's rho: Pearson's r of the average ranks, finite pairs only."""
    x, y = _finite_pairs(x, y)
    if len(x) < 2:
        return math.nan
    return pearson(_ranks(x), _ranks(y))


def correlation(x, y):
    """Spearman and Pearson with the number of finite pairs they use."""
    n = len(_finite_pairs(x, y)[0])
    return {"spearman": spearman(x, y), "pearson": pearson(x, y), "n": n}
