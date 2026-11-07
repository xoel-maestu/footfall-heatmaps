import numpy as np
import pytest

from space_usage_heatmaps.occupancy.domain.association import (
    Association,
    CameraTracks,
    associate,
    merge_points,
    mutual_nearest,
)

FPS = 10.0


def track(tid, frames, x, y, noise=0.0, seed=0):
    frames = np.asarray(list(frames))
    rng = np.random.default_rng(seed)
    xs = np.broadcast_to(np.asarray(x, float), frames.shape) + rng.normal(0, noise, len(frames))
    ys = np.broadcast_to(np.asarray(y, float), frames.shape) + rng.normal(0, noise, len(frames))
    return np.full(len(frames), tid), frames, np.column_stack([xs, ys])


def camera(name, *tracks):
    ids, frames, xy = (np.concatenate(p) for p in zip(*tracks, strict=True))
    return CameraTracks(name, ids, frames, xy)


def walk(frames):
    """A person walking along x at 1 m/s."""
    f = np.asarray(list(frames))
    return f / FPS, 0.0 * f


def same(gids_a, gids_b):
    return len(set(gids_a) | set(gids_b)) == 1


def test_without_association_every_camera_track_is_its_own_id():
    a = camera("a", track(1, range(50), 0, 0), track(2, range(50), 5, 0))
    b = camera("b", track(1, range(50), 0, 0))

    ga, gb = associate([a, b], Association("none"), FPS)

    assert len(set(ga) | set(gb)) == 3
    assert len(set(ga[a.track_ids == 1])) == 1


@pytest.mark.parametrize("method", ["colocate", "colocate-mnn"])
def test_two_cameras_seeing_one_person_share_an_id_and_a_far_person_does_not(method):
    a = camera("a", track(1, range(50), *walk(range(50)), noise=0.1, seed=1))
    b = camera(
        "b",
        track(7, range(50), *walk(range(50)), noise=0.1, seed=2),
        track(8, range(50), 0, 6.0),
    )

    ga, gb = associate([a, b], Association(method), FPS)

    assert same(ga, gb[b.track_ids == 7])
    assert set(gb[b.track_ids == 8]).isdisjoint(ga)


def test_fragments_of_one_camera_join_through_another_cameras_continuous_track():
    a = camera(
        "a",
        track(1, range(0, 40), *walk(range(0, 40))),
        track(2, range(45, 90), *walk(range(45, 90))),
    )
    b = camera("b", track(5, range(0, 90), *walk(range(0, 90))))

    ga, gb = associate([a, b], Association("colocate"), FPS)

    assert len(set(ga)) == 1 and same(ga, gb)


def test_two_concurrent_tracks_of_one_camera_never_share_an_id():
    # Two people side by side, 0.4 m apart: both within the gate of camera b's track.
    a = camera("a", track(1, range(60), 0, 0.0), track(2, range(60), 0, 0.4))
    b = camera("b", track(5, range(60), 0, 0.2))

    ga, _ = associate([a, b], Association("colocate"), FPS)

    assert ga[a.track_ids == 1][0] != ga[a.track_ids == 2][0]


def test_tracks_seen_together_in_different_places_are_not_joined_through_a_third():
    # a1 and b1 are seen together 3 m apart; c's one track (an id switch) sits
    # on a1 for half the run, then on b1: it co-locates with both.
    a = camera("a", track(1, range(0, 100), 0, 0))
    b = camera("b", track(1, range(0, 100), 3, 0))
    c = camera("c", (np.ones(100, int), np.arange(100), np.repeat([[0.0, 0], [3, 0]], 50, axis=0)))

    ga, gb, gc = associate([a, b, c], Association("colocate"), FPS)

    assert ga[0] != gb[0]
    assert gc[0] in (ga[0], gb[0])


def test_a_short_co_location_is_not_enough():
    a = camera("a", track(1, range(0, 5), 0, 0))
    b = camera("b", track(1, range(0, 100), 0, 0))

    ga, gb = associate([a, b], Association("colocate", min_overlap_s=1.0), FPS)

    assert ga[0] != gb[0]


def test_handover_joins_a_track_that_starts_where_another_ended():
    a = camera("a", track(1, range(0, 40), *walk(range(0, 40))))
    b = camera("b", track(1, range(45, 90), *walk(range(45, 90))))
    far = camera("c", track(1, range(45, 90), 20.0, 0))

    plain = associate([a, b, far], Association("colocate"), FPS)
    handed = associate([a, b, far], Association("colocate", handover_s=2.0), FPS)

    assert plain[0][0] != plain[1][0]
    assert handed[0][0] == handed[1][0]
    assert handed[2][0] not in (handed[0][0],)


def test_handover_respects_the_gap_and_the_distance():
    a = camera("a", track(1, range(0, 40), 0, 0))
    late = camera("b", track(1, range(80, 120), 0, 0))  # 4 s later
    jump = camera("c", track(1, range(45, 90), 6.0, 0))  # 6 m in 0.5 s

    ga, gl, gj = associate([a, late, jump], Association("colocate", handover_s=2.0), FPS)

    assert len({ga[0], gl[0], gj[0]}) == 3


def test_handover_never_joins_two_tracks_of_one_camera_alive_together():
    # Track 1's floor points end at frame 39, but the tracker keeps it alive with
    # no floor point up to frame 60; track 2 of the same camera starts at 45 on
    # the same spot. They are two people at once, never one.
    ids, frames, xy = track(1, range(0, 61), 0.0, 0.0)
    xy[40:] = np.nan
    a = camera("a", (ids, frames, xy), track(2, range(45, 90), 0.0, 0.0))
    fragment = camera("b", track(1, range(0, 40), 9.0, 0), track(2, range(45, 90), 9.0, 0))

    ga, gb = associate([a, fragment], Association("colocate", handover_s=2.0), FPS)

    assert ga[0] != ga[-1]
    assert gb[0] == gb[-1]  # a fragment of one camera's track still continues


def test_merged_points_are_one_per_id_and_frame_at_the_mean():
    a = camera("a", track(1, [0, 1], 0.0, 0.0))
    b = camera("b", track(1, [1, 2], 1.0, 0.0))
    nan = CameraTracks("c", np.array([3]), np.array([1]), np.array([[np.nan, np.nan]]))
    gids = [np.array([10, 10]), np.array([10, 10]), np.array([10])]

    ids, frames, xy = merge_points([a, b, nan], gids)

    order = np.argsort(frames)
    np.testing.assert_array_equal(ids[order], [10, 10, 10])
    np.testing.assert_array_equal(frames[order], [0, 1, 2])
    np.testing.assert_allclose(xy[order], [[0, 0], [0.5, 0], [1, 0]])


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="method"):
        Association("appearance")


def test_mutual_nearest_pairs_each_point_with_one_point_per_other_camera():
    # Camera 0: points at x 0 and 0.5; camera 1: one point at x 0.45.
    xy = np.array([[0.0, 0], [0.5, 0], [0.45, 0]])
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)

    m = mutual_nearest(d, np.array([0, 0, 1]))

    assert m[1, 2] and m[2, 1]
    assert not m[0, 2] and not m[2, 0]
