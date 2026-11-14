import numpy as np
import pytest
from trackers.eval import compute_identity_metrics

from footfall_heatmaps.evaluation.infrastructure.floor_identity import floor_identity


def _points(rng, n_frames, n_ids, jitter):
    ids, frames, xy = [], [], []
    for f in range(n_frames):
        present = rng.random(n_ids) < 0.8
        for i in np.flatnonzero(present):
            ids.append(i)
            frames.append(f)
            xy.append([i * 1.5 + rng.normal(0, jitter), f * 0.05 + rng.normal(0, jitter)])
    return np.array(ids), np.array(frames), np.array(xy, float).reshape(-1, 2)


def _reference(t_ids, t_frames, t_xy, p_ids, p_frames, p_xy, gate):
    """TrackEval's IDF1 with its Euclidean similarity, 1 - d / (2 gate), at 0.5."""
    frames = np.union1d(t_frames, p_frames)
    gt, tr, sim = [], [], []
    for f in frames:
        a, b = t_frames == f, p_frames == f
        d = np.linalg.norm(t_xy[a][:, None] - p_xy[b][None], axis=2)
        gt.append(t_ids[a])
        tr.append(p_ids[b])
        sim.append(np.clip(1 - d / (2 * gate), 0, None))
    return compute_identity_metrics(gt, tr, sim)


def test_it_equals_trackeval_idf1_with_a_floor_distance_gate():
    rng = np.random.default_rng(3)
    truth = _points(rng, 60, 6, 0.0)
    p_ids, p_frames, p_xy = _points(rng, 60, 6, 0.4)
    p_ids = p_ids * 10 + (p_frames // 17)  # identity switches every 17 frames
    ref = _reference(*truth, p_ids, p_frames, p_xy, 1.0)

    r = floor_identity(*truth, p_ids, p_frames, p_xy, gate_m=1.0, blocks=p_frames.max() + 1)

    assert r.idtp.sum() == ref["IDTP"]
    assert r.idf1 == pytest.approx(ref["IDF1"])


def test_per_block_counts_add_up_to_the_whole_run():
    rng = np.random.default_rng(4)
    truth = _points(rng, 40, 4, 0.0)
    pred = _points(rng, 40, 4, 0.3)
    block = lambda f: f // 10  # noqa: E731

    r = floor_identity(*truth, *pred, gate_m=1.0, blocks=4, block_of=block)

    assert r.idtp.shape == r.n_truth.shape == r.n_pred.shape == (4,)
    assert r.n_truth.sum() == len(truth[0]) and r.n_pred.sum() == len(pred[0])
    np.testing.assert_array_equal(r.n_truth, np.bincount(truth[1] // 10, minlength=4))
    assert 2 * r.idtp.sum() / (r.n_truth.sum() + r.n_pred.sum()) == pytest.approx(r.idf1)


def test_a_prediction_beyond_the_gate_is_no_match():
    one = (np.array([1]), np.array([0]), np.array([[0.0, 0.0]]))
    far = (np.array([7]), np.array([0]), np.array([[1.01, 0.0]]))
    near = (np.array([7]), np.array([0]), np.array([[0.99, 0.0]]))

    assert floor_identity(*one, *far, gate_m=1.0, blocks=1).idf1 == 0.0
    assert floor_identity(*one, *near, gate_m=1.0, blocks=1).idf1 == 1.0


def test_a_prediction_id_twice_in_one_frame_is_refused():
    one = (np.array([1]), np.array([0]), np.array([[0.0, 0.0]]))
    twice = (np.array([7, 7]), np.array([0, 0]), np.array([[0.0, 0.0], [3.0, 0.0]]))

    with pytest.raises(ValueError, match="twice"):
        floor_identity(*one, *twice, gate_m=1.0, blocks=1)
