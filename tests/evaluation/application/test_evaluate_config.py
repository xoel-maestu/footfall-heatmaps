import numpy as np
import pytest

from space_usage_heatmaps.evaluation.application.evaluate_config import evaluate_config
from space_usage_heatmaps.evaluation.domain.run_config import RunConfig
from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from tests.conftest import write_scene

CFG = RunConfig("yolov8n", "bytetrack", 640, "bottom", "ultralytics", 1)


class InMemoryTrackSource:
    """Tracks given per (scene, camera)."""

    def __init__(self, tracks):
        self._tracks = tracks
        self.calls = []

    def tracks(self, scene, camera):
        self.calls.append((scene, camera))
        return self._tracks[(scene, camera)]


def tracks_from_rows(rows, camera=7):
    r = np.array([row for row in rows if row[0] == camera])
    x, y, w, h = r[:, 3:7].T
    return Tracks(r[:, 1].astype(int), r[:, 2].astype(int), np.stack([x, y, x + w, y + h], 1))


@pytest.fixture
def reader(tmp_path, rows):
    return NvidiaMtmcReader(write_scene(tmp_path, rows))


def test_true_boxes_at_their_foot_point_score_zero_on_the_visible_truth(reader, rows):
    source = InMemoryTrackSource({("001", "7"): tracks_from_rows(rows)})

    report = evaluate_config(reader, [("001", "7")], [(CFG, source)], cell=0.5, n_resamples=50)

    for layer in ("dwell", "passage"):
        e = report.results["visible@coverage"].scenes[0][0][layer]["tvd"]
        assert (e.value, e.lo, e.hi) == pytest.approx((0, 0, 0), abs=1e-12)
        assert report.results["visible@coverage"].pooled[0][layer]["mass_error"].value == 0
    assert report.scenes[0].samples == 15 and report.scenes[0].dropped == 0
    assert list(report.results) == ["visible@coverage", "full@coverage", "visible@visibility"]


def test_compare_gives_the_paired_difference_of_the_two_configs(reader, rows):
    source = InMemoryTrackSource({("001", "7"): tracks_from_rows(rows)})
    centroid = RunConfig(**(vars(CFG) | {"foot_point": "centroid"}))

    report = evaluate_config(
        reader, [("001", "7")], [(centroid, source), (CFG, source)], cell=0.5, n_resamples=50
    )

    # The box centre is 10 px above the feet: it lands further away on the floor.
    vc = report.results["visible@coverage"]
    assert vc.scenes[0][0]["dwell"]["tvd"].value > 0
    assert vc.delta_pooled["dwell"]["tvd"].value == vc.scenes[0][0]["dwell"]["tvd"].value


def test_each_sample_is_one_frame_whatever_the_stride(reader, rows):
    """The pipeline fills the frames a stride skips (fill_stride), so its tracks
    have one sample per source frame and dwell counts each at 1 / fps."""
    strided = RunConfig(**(vars(CFG) | {"stride": 2, "engine": "onnx-cuda"}))
    source = InMemoryTrackSource({("001", "7"): tracks_from_rows(rows)})

    report = evaluate_config(reader, [("001", "7")], [(strided, source)], cell=0.5, n_resamples=10)

    assert report.results["visible@coverage"].scenes[0][0]["dwell"]["mass_error"].value == 0


def test_predictions_outside_the_truth_frames_are_not_scored(reader, rows):
    t = tracks_from_rows(rows)
    late = Tracks(
        np.append(t.track_ids, 1), np.append(t.frames, 500), np.vstack([t.boxes, t.boxes[:1]])
    )
    source = InMemoryTrackSource({("001", "7"): late})

    report = evaluate_config(reader, [("001", "7")], [(CFG, source)], cell=0.5, n_resamples=10)

    assert report.scenes[0].samples == 15 and report.scenes[0].outside_truth_frames == 1
    assert report.results["visible@coverage"].scenes[0][0]["dwell"]["tvd"].value == 0


def test_head_without_a_camera_matrix_is_an_error_and_bottom_still_works(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows, extrinsics=False))
    source = InMemoryTrackSource({("001", "7"): tracks_from_rows(rows)})
    head = RunConfig(**(vars(CFG) | {"foot_point": "head"}))

    with pytest.raises(ValueError, match="camera matrix"):
        evaluate_config(reader, [("001", "7")], [(head, source)], cell=0.5, n_resamples=5)
    report = evaluate_config(reader, [("001", "7")], [(CFG, source)], cell=0.5, n_resamples=5)
    assert report.results["visible@coverage"].pooled[0]["dwell"]["tvd"].value == 0
