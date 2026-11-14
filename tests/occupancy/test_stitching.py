import numpy as np
import pytest

from footfall_heatmaps.occupancy.domain.stitching import Stitch, stitch

FPS = 10.0
S = Stitch(gap_s=2.0, radius_m=0.5)  # v_max 1.5 m/s


def walk(tid, frames, x0=0.0, y0=0.0, vx=1.0, vy=0.0):
    """Track tid walking at (vx, vy) m/s from (x0, y0) at its first frame."""
    f = np.asarray(list(frames))
    t = (f - f[0]) / FPS
    return np.full(len(f), tid), f, np.column_stack([x0 + vx * t, y0 + vy * t])


def still(tid, frames, x, y):
    return walk(tid, frames, x, y, 0.0, 0.0)


def rows(*tracks):
    ids, frames, xy = (np.concatenate(p) for p in zip(*tracks, strict=True))
    return ids, frames, xy


def groups(ids, new):
    """The old ids each new id holds."""
    return sorted(sorted({int(i) for i in ids[new == n]}) for n in np.unique(new))


def test_a_track_resumed_after_a_short_gap_where_it_was_heading_is_joined():
    a = walk(1, range(0, 50))  # ends at x 4.9 at frame 49
    b = walk(2, range(65, 100), x0=6.4)  # 1.6 s later, where 1 m/s puts it

    ids, frames, xy = rows(a, b)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1, 2]]


def test_a_gap_longer_than_gap_s_is_not_joined():
    a = walk(1, range(0, 50))
    b = walk(2, range(71, 100), x0=7.0)  # 2.2 s later, on the predicted path

    ids, frames, xy = rows(a, b)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1], [2]]


def test_tracks_alive_in_the_same_frame_are_never_joined():
    a = walk(1, range(0, 50))
    b = walk(2, range(49, 100), x0=4.9)  # starts on a's last frame, at its spot

    ids, frames, xy = rows(a, b)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1], [2]]


def test_a_start_beyond_radius_plus_v_max_times_gap_of_the_prediction_is_not_joined():
    a = still(1, range(0, 50), 0.0, 0.0)
    # 1.1 s gap: reach 0.5 + 1.5 x 1.1 = 2.15 m from the prediction (0, 0).
    near = still(2, range(60, 100), 0.0, 2.1)
    far = still(3, range(60, 100), 0.0, -2.2)

    ids, frames, xy = rows(a, near)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1, 2]]
    ids, frames, xy = rows(a, far)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1], [3]]


def test_the_prediction_follows_the_track_velocity():
    a = walk(1, range(0, 50))  # heading +x, ends at 4.9
    ahead = still(2, range(60, 100), 5.9 + 0.3, 0.0)  # 0.3 m from the prediction 5.9
    behind = still(3, range(60, 100), 4.9 - 0.4, 0.0)  # 0.4 m from the end, 1.4 m from it

    ids, frames, xy = rows(a, ahead, behind)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1, 2], [3]]


def test_an_end_with_two_starts_in_reach_takes_the_nearer_one_only():
    a = still(1, range(0, 50), 0.0, 0.0)
    b = still(2, range(60, 100), 0.0, 0.3)  # 0.3 m from a's end
    c = still(3, range(60, 100), 0.0, -1.0)  # 1.0 m, also within reach 2.15 m

    ids, frames, xy = rows(a, b, c)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1, 2], [3]]


def test_two_ends_competing_for_one_start_give_it_to_the_nearer_one():
    a = still(1, range(0, 50), 0.0, 0.0)
    b = still(2, range(0, 50), 0.0, 1.0)
    c = still(3, range(60, 100), 0.0, 0.8)  # 0.8 m from a's end, 0.2 m from b's

    ids, frames, xy = rows(a, b, c)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1], [2, 3]]


def test_a_chain_of_fragments_gets_one_id():
    a = walk(1, range(0, 30))
    b = walk(2, range(40, 70), x0=4.0)
    c = walk(3, range(80, 110), x0=8.0)

    ids, frames, xy = rows(a, b, c)
    new = stitch(ids, frames, xy, FPS, S)
    assert groups(ids, new) == [[1, 2, 3]]
    assert set(new.tolist()) == {1}  # the chain keeps its first track's id


def test_stitching_only_relabels_same_rows_in_and_out():
    rng = np.random.default_rng(0)
    tracks = [
        walk(k, range(s, s + n), x0=x, vx=v)
        for k, (s, n, x, v) in enumerate(
            zip(
                rng.integers(0, 300, 25),
                rng.integers(5, 60, 25),
                rng.uniform(-5, 5, 25),
                rng.uniform(-1.5, 1.5, 25),
                strict=True,
            )
        )
    ]
    ids, frames, xy = rows(*tracks)
    order = rng.permutation(len(ids))  # rows in any order
    ids, frames, xy = ids[order], frames[order], xy[order]
    xy[::7] = np.nan  # boxes that could not be placed
    before = (ids.copy(), frames.copy(), xy.copy())

    new = stitch(ids, frames, xy, FPS, Stitch(gap_s=10.0, radius_m=1.0))

    assert new.shape == ids.shape and new.dtype == ids.dtype
    for got, was in zip((ids, frames, xy), before, strict=True):
        np.testing.assert_array_equal(got, was)
    # Every old track lands whole in one new id, and no new id is alive twice in a frame.
    for k in np.unique(ids):
        assert len(np.unique(new[ids == k])) == 1
    for n in np.unique(new):
        f = frames[new == n]
        assert len(np.unique(f)) == len(f)
    assert len(np.unique(new)) < len(np.unique(ids))


def test_a_track_without_a_floor_point_is_left_alone():
    a = walk(1, range(0, 50))
    b = walk(2, range(65, 100), x0=6.4)
    b = (b[0], b[1], np.full_like(b[2], np.nan))

    ids, frames, xy = rows(a, b)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1], [2]]


def test_no_rows_give_no_ids():
    empty = np.array([], dtype=np.int64)
    assert stitch(empty, empty, np.empty((0, 2)), FPS, S).shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap_s": 0.0, "radius_m": 0.5},
        {"gap_s": 2.0, "radius_m": -0.1},
        {"gap_s": 2.0, "radius_m": 0.5, "v_max_mps": -1.0},
        {"gap_s": 2.0, "radius_m": 0.5, "velocity_window_s": 0.0},
    ],
)
def test_invalid_parameters_are_errors(kwargs):
    with pytest.raises(ValueError, match="stitch"):
        Stitch(**kwargs)


def test_the_predicted_speed_is_capped_at_v_max():
    a = still(1, range(48, 50), 0.0, 0.0)
    a[2][-1] = [1.0, 0.0]  # two frames, a jump of 1 m: 10 m/s
    # 1.1 s gap: capped at 1.5 m/s the prediction is x 2.65; at 10 m/s, x 12.
    b = still(2, range(60, 100), 2.65, 0.0)

    ids, frames, xy = rows(a, b)
    assert groups(ids, stitch(ids, frames, xy, FPS, S)) == [[1, 2]]
