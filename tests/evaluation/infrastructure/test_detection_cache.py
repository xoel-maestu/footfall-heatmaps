import json

import numpy as np

from space_usage_heatmaps.evaluation.infrastructure.detection_cache import CachedDetections
from space_usage_heatmaps.perception.domain.frame import Detections

META = {"weights_sha256": "ab" * 32, "input_size": 960, "score_thr": 0.1}
# Three processed frames; the middle one has no box, and must still be replayed.
RUN = [
    (0, Detections(np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]), np.array([0.9, 0.2]))),
    (1, Detections(np.empty((0, 4)), np.empty(0))),
    (2, Detections(np.array([[2.0, 3, 4, 5]]), np.array([0.7]))),
]


class CountingDetection:
    """The detector's source on a cache miss: counts the detector runs."""

    def __init__(self, providers=None):
        self.runs = 0
        self.last_run = {"providers": providers} if providers else None

    def detections(self, scene, camera):
        self.runs += 1
        return iter(RUN)


def assert_same(got, want=RUN):
    assert [i for i, _ in got] == [i for i, _ in want]
    for (_, a), (_, b) in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.boxes.reshape(-1, 4), b.boxes)
        np.testing.assert_array_equal(a.scores, b.scores)


def test_a_miss_runs_the_detector_once_and_a_hit_replays_every_frame(tmp_path):
    run = CountingDetection()

    first = CachedDetections(run, tmp_path, META).detections("071", "645")
    again = CachedDetections(run, tmp_path, dict(META)).detections("071", "645")

    assert run.runs == 1
    assert_same(first)
    assert_same(again)


def test_the_file_holds_boxes_scores_and_frame_indices_only(tmp_path):
    CachedDetections(CountingDetection(), tmp_path, META).detections("071", "645")

    (path,) = tmp_path.iterdir()
    assert path.name == "scene_071_camera_645.npz"
    with np.load(path, allow_pickle=False) as f:
        assert sorted(f.files) == ["box_frames", "boxes", "frames", "meta", "scores"]
        assert f["frames"].tolist() == [0, 1, 2] and f["box_frames"].tolist() == [0, 0, 2]


def test_another_detector_recipe_recomputes_and_says_why(tmp_path, capsys):
    run = CountingDetection()
    CachedDetections(run, tmp_path, META).detections("071", "645")

    CachedDetections(run, tmp_path, META | {"input_size": 640}).detections("071", "645")

    assert run.runs == 2
    assert "input_size 960 != 640" in capsys.readouterr().err


def test_the_providers_the_detector_ran_on_are_stored_and_replayed(tmp_path):
    run = CountingDetection(providers=["CUDAExecutionProvider"])
    CachedDetections(run, tmp_path, META).detections("071", "645")

    cached = CachedDetections(CountingDetection(), tmp_path, META)
    cached.detections("071", "645")

    assert cached.last_run == {"providers": ["CUDAExecutionProvider"]}
    with np.load(tmp_path / "scene_071_camera_645.npz") as f:
        assert json.loads(str(f["meta"]))["run"] == {"providers": ["CUDAExecutionProvider"]}


def test_a_corrupt_file_or_one_without_metadata_is_recomputed(tmp_path, capsys):
    run = CountingDetection()
    path = tmp_path / "scene_071_camera_645.npz"
    path.write_bytes(b"not a zip file")
    assert_same(CachedDetections(run, tmp_path, META).detections("071", "645"))

    empty = np.empty(0)
    np.savez(path, frames=empty, box_frames=empty, boxes=empty, scores=empty)  # no meta key
    assert_same(CachedDetections(run, tmp_path, META).detections("071", "645"))

    assert run.runs == 2
    err = capsys.readouterr().err
    assert "unreadable" in err and "no metadata" in err


def test_another_video_recomputes(tmp_path, capsys):
    run = CountingDetection()
    CachedDetections(run, tmp_path, META, identity=lambda s, c: {"video_bytes": 7}).detections(
        "071", "645"
    )

    CachedDetections(run, tmp_path, META, identity=lambda s, c: {"video_bytes": 8}).detections(
        "071", "645"
    )

    assert run.runs == 2
    assert "video_bytes 7 != 8" in capsys.readouterr().err


TRT = {"engine": "onnx-trt"}


def test_detections_after_a_tensorrt_fallback_are_not_stored(tmp_path, capsys):
    run = CountingDetection(["CUDAExecutionProvider"])

    got = CachedDetections(run, tmp_path, META | TRT).detections("071", "645")
    CachedDetections(run, tmp_path, META | TRT).detections("071", "645")

    assert_same(got)
    assert run.runs == 2
    assert not list(tmp_path.glob("*.npz"))
    assert "not TensorRT" in capsys.readouterr().err


def test_detections_that_ran_on_tensorrt_are_stored(tmp_path):
    run = CountingDetection(["TensorrtExecutionProvider"])

    CachedDetections(run, tmp_path, META | TRT).detections("071", "645")
    CachedDetections(run, tmp_path, META | TRT).detections("071", "645")

    assert run.runs == 1
