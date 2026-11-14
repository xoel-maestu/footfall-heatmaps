import os
from pathlib import Path

import numpy as np
import pytest

from footfall_heatmaps.perception.domain.frame import Frame
from footfall_heatmaps.perception.infrastructure.onnx_dfine_detector import (
    PERSON,
    SCORE_THR,
    OnnxDfineDetector,
    postprocess,
    preprocess,
)


def test_preprocess_letterboxes_centred_on_black_in_rgb_0_to_1():
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 0] = 255  # blue in OpenCV's BGR

    x, ratio, (pad_w, pad_h) = preprocess(image, 64)

    assert ratio == pytest.approx(0.32) and (pad_w, pad_h) == (0, 16)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    assert x[2, 32, 32] == pytest.approx(1.0) and x[0, 32, 32] == 0  # RGB: blue is last
    assert (x[:, :16] == 0).all() and (x[:, 48:] == 0).all()  # 32 image rows, centred


def test_postprocess_keeps_persons_above_the_threshold_in_source_pixels():
    labels = np.array([[PERSON, 2, PERSON, PERSON]])
    boxes = np.array([[[10, 26, 20, 46], [0, 0, 5, 5], [0, 16, 8, 24], [1, 1, 2, 2]]], np.float32)
    scores = np.array([[0.9, 0.95, 0.5, SCORE_THR]], np.float32)

    d = postprocess(labels, boxes, scores, 0.5, (0, 16), (100, 200))

    np.testing.assert_allclose(d.boxes, [[20, 20, 40, 60], [0, 0, 16, 16]])
    np.testing.assert_allclose(d.scores, [0.9, 0.5])


def test_postprocess_clips_to_the_image():
    d = postprocess(
        np.array([[PERSON]]), np.array([[[-4, 0, 70, 40]]], np.float32), np.array([[0.8]]),
        0.5, (0, 0), (60, 100),
    )  # fmt: skip

    np.testing.assert_allclose(d.boxes, [[0, 0, 100, 60]])


class FakeSession:
    def __init__(self, size=64):
        self.size, self.feeds = size, []

    def get_inputs(self):
        return [type("I", (), {"name": "images", "shape": ["N", 3, self.size, self.size]})]

    def run(self, _, feed):
        self.feeds.append(feed)
        # one person at (16, 16)-(48, 48) of the letterboxed input
        return [np.array([[PERSON]]), np.array([[[16, 16, 48, 48]]], np.float32), np.array([[0.7]])]


def test_detect_feeds_the_input_size_as_target_size_and_maps_back():
    session = FakeSession()

    d = OnnxDfineDetector(session).detect(Frame(0, 0.0, np.zeros((128, 128, 3), np.uint8)))

    feed = session.feeds[0]
    assert feed["images"].shape == (1, 3, 64, 64)
    assert feed["orig_target_sizes"].tolist() == [[64, 64]]
    assert feed["orig_target_sizes"].dtype == np.int64
    np.testing.assert_allclose(d.boxes, [[32, 32, 96, 96]])


def test_the_exported_model_finds_people_in_a_real_frame():
    """Adapter against the exported D-FINE-S; needs the runtime group, the export
    (scripts/export-detector.py dfine-s) and scene 071's video."""
    import cv2

    from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import (
        NvidiaMtmcReader,
    )

    pytest.importorskip("onnxruntime", reason="runtime group not installed")
    data = Path(os.environ.get("FOOTFALL_DATA_DIR", "~/datasets/footfall-heatmaps"))
    data = data.expanduser()
    weights = data / "models" / "dfine_s_obj2coco.onnx"
    video = NvidiaMtmcReader(data).video("071", "645")
    if not weights.exists() or not video.exists():
        pytest.skip("no D-FINE-S export or no scene 071 video")
    cap = cv2.VideoCapture(str(video))
    ok, image = cap.read()  # in memory only
    cap.release()
    padded = np.zeros((1080 + 840, 1920, 3), np.uint8)  # square: no letterbox bars
    padded[420 : 420 + 1080] = image
    detector = OnnxDfineDetector.load(weights)

    a = detector.detect(Frame(0, 0.0, image))
    b = detector.detect(Frame(0, 0.0, padded))

    assert detector.input_size == 640
    assert (a.scores > 0.5).sum() >= 3  # people in the first frame of scene 071
    # The same frame letterboxed by the adapter or padded by hand: same people, same place.
    strong_a = a.boxes[a.scores > 0.5]
    strong_b = b.boxes[b.scores > 0.5] - [0, 420, 0, 420]
    order_a, order_b = np.lexsort(strong_a.T[::-1]), np.lexsort(strong_b.T[::-1])
    assert len(strong_a) == len(strong_b)
    np.testing.assert_allclose(strong_b[order_b], strong_a[order_a], atol=3)
