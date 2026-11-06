"""UCMCTrack-style association on the floor, on a synthetic camera.

The camera of tests/conftest.py: at the origin looking along +y, floor point
(x, y) -> pixel u = (100 x + 50 y) / y, v = (50 y + 100) / y. The horizon is
v = 50; a box whose bottom is above it does not touch the floor.
"""

import numpy as np

from space_usage_heatmaps.perception.infrastructure.ucmc_tracker import (
    UcmcTracker,
    floor_points,
)

H = np.array([[100.0, 50.0, 0.0], [0.0, 50.0, 100.0], [0.0, 1.0, 0.0]])
FPS = 30.0


def box_at(x, y, height_px=40.0, width_px=16.0):
    """Box whose bottom-centre is the image of floor point (x, y)."""
    u, v = (100 * x + 50 * y) / y, (50 * y + 100) / y
    return [u - width_px / 2, v - height_px, u + width_px / 2, v]


def walk(x0, y0, frame, vx=1.2):
    return box_at(x0 + vx * frame / FPS, y0)


def run(tracker, boxes_per_frame, score=0.9):
    out = []
    for i, boxes in enumerate(boxes_per_frame):
        b = np.array(boxes, float).reshape(-1, 4)
        out.append(tracker.update(b, np.full(len(b), score), i))
    return out


def ids_of(out):
    return [ids.tolist() for ids, _ in out]


def test_floor_points_invert_the_homography_with_its_jacobian():
    boxes = np.array([box_at(0.5, 4.0), box_at(-1.0, 8.0)])

    xy, cov, front = floor_points(boxes, np.linalg.inv(H))

    np.testing.assert_allclose(xy, [[0.5, 4.0], [-1.0, 8.0]], atol=1e-9)
    assert front.all()
    # pixel noise 2 px (0.05 x 40 px, clipped up to 2) through the numeric Jacobian
    for (x, y), c in zip(xy, cov, strict=True):
        eps = 1e-6
        uv = np.array([(100 * x + 50 * y) / y, (50 * y + 100) / y])

        def to_floor(p):
            w = np.linalg.inv(H) @ np.array([p[0], p[1], 1.0])
            return w[:2] / w[2]

        jac = np.column_stack(
            [(to_floor(uv + d) - to_floor(uv - d)) / (2 * eps) for d in np.eye(2) * eps]
        )
        np.testing.assert_allclose(c, jac @ np.diag([4.0, 4.0]) @ jac.T, rtol=1e-4)


def test_a_box_above_the_horizon_is_not_on_the_floor():
    boxes = np.array([[90.0, 0.0, 110.0, 40.0]])  # bottom v = 40 < horizon 50

    _, _, front = floor_points(boxes, np.linalg.inv(H))

    assert not front.any()
    ids, out = UcmcTracker(FPS, H).update(boxes, np.array([0.9]), 0)
    assert ids.shape == (0,) and out.shape == (0, 4)


def test_a_walking_person_keeps_one_id_from_its_third_frame():
    out = run(UcmcTracker(FPS, H), [[walk(0.0, 5.0, f)] for f in range(30)])

    assert ids_of(out)[:2] == [[], []]  # tentative on its first two frames
    assert all(ids == [0] for ids in ids_of(out)[2:])
    np.testing.assert_allclose(out[10][1], [walk(0.0, 5.0, 10)])  # the detection's box


def test_two_people_get_two_ids():
    out = run(UcmcTracker(FPS, H), [[walk(-1.0, 5.0, f), walk(1.0, 7.0, f)] for f in range(5)])

    assert sorted(out[4][0].tolist()) == [0, 1]


def test_a_short_gap_keeps_the_id_and_a_long_one_does_not():
    short = [[walk(0.0, 5.0, f)] if not 10 <= f < 20 else [] for f in range(40)]
    long = [[walk(0.0, 5.0, f)] if not 10 <= f < 50 else [] for f in range(60)]

    assert {i for ids in ids_of(run(UcmcTracker(FPS, H), short)) for i in ids} == {0}
    # cdt 30 frames: the coasting track is dropped and the person comes back new
    assert {i for ids in ids_of(run(UcmcTracker(FPS, H), long)) for i in ids} == {0, 1}


def test_a_lost_track_coasts_even_with_no_low_score_box():
    """The original only marks a track coasted inside its low-score stage, so a
    track missed in a frame without low-score boxes is never deleted. Here it is."""
    tracker = UcmcTracker(FPS, H)
    run(tracker, [[walk(0.0, 5.0, f)] for f in range(5)] + [[] for _ in range(40)])

    assert tracker._tracks == []


def test_scores_between_the_two_thresholds_keep_a_track_but_never_start_one():
    tracker = UcmcTracker(FPS, H)
    start = run(tracker, [[walk(0.0, 5.0, f)] for f in range(5)])
    kept = [tracker.update(np.array([walk(0.0, 5.0, f)]), np.array([0.55]), f) for f in range(5, 8)]

    assert ids_of(start)[-1] == [0] and all(ids.tolist() == [0] for ids, _ in kept)
    assert (
        ids_of(run(UcmcTracker(FPS, H), [[walk(0.0, 5.0, f)] for f in range(5)], 0.55)) == [[]] * 5
    )
    assert (
        ids_of(run(UcmcTracker(FPS, H), [[walk(0.0, 5.0, f)] for f in range(5)], 0.4)) == [[]] * 5
    )


def test_no_boxes_gives_empty_arrays():
    ids, boxes = UcmcTracker(FPS, H).update(np.empty((0, 4)), np.empty(0), 0)

    assert ids.shape == (0,) and boxes.shape == (0, 4)


def test_parameters_are_the_published_defaults_but_the_frame_rate():
    p = UcmcTracker(25.0, H).parameters

    assert p == {
        "wx": 0.1,
        "wy": 0.1,
        "vmax": 0.5,
        "a1": 10.0,
        "a2": 10.0,
        "cdt": 30,
        "high_score": 0.6,
        "conf_thresh": 0.5,
        "frame_rate": 25.0,
    }


def test_a_lower_conf_thresh_lets_low_score_boxes_continue_a_track():
    frames = [[walk(0.0, 5.0, f)] for f in range(10)]
    scores = [0.9] * 5 + [0.3] * 5

    def last_ids(params):
        t = UcmcTracker(FPS, H, params)
        out = [
            t.update(np.array(b), np.array([s]), i)
            for i, (b, s) in enumerate(zip(frames, scores, strict=True))
        ]
        return [ids.tolist() for ids, _ in out[5:]]

    assert last_ids(None) == [[]] * 5  # default conf_thresh 0.5 drops them
    assert last_ids({"conf_thresh": 0.1}) == [[0]] * 5
