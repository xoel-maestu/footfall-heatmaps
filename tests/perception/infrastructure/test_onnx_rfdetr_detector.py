import os
from pathlib import Path

import numpy as np
import pytest

from footfall_heatmaps.perception.domain.frame import Frame
from footfall_heatmaps.perception.infrastructure.onnx_rfdetr_detector import (
    MEAN,
    PERSON,
    SCORE_THR,
    STD,
    OnnxRfdetrDetector,
    postprocess,
    preprocess,
)


def test_preprocess_stretches_to_the_square_input_in_rgb_with_imagenet_statistics():
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 0] = 255  # blue in OpenCV's BGR

    x = preprocess(image, 64)

    assert x.shape == (3, 64, 64) and x.dtype == np.float32  # no letterbox: stretched
    np.testing.assert_allclose(x[:, 10, 10], (np.array([0, 0, 1]) - MEAN) / STD, rtol=1e-6)


def test_preprocess_resizes_bilinearly_with_half_pixel_centres_and_no_antialias():
    image = np.zeros((2, 4, 3), np.uint8)
    image[:, 1::2] = 255  # columns alternate black and white

    x = preprocess(image, 2)

    # Half-pixel centres: output column 0 samples input x = 0.5, halfway between 0 and 255.
    rgb = x * STD[:, None, None] + MEAN[:, None, None]
    np.testing.assert_allclose(rgb[:, :, 0], 0.5, atol=1e-6)


def logit(p):
    return np.log(p / (1 - p))


def test_postprocess_keeps_person_query_class_pairs_above_the_threshold():
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 1.0, 1.0]]])
    logits = np.full((1, 3, 91), -20.0)
    logits[0, 0, PERSON] = logit(0.9)
    logits[0, 1, 3] = logit(0.95)  # a car
    logits[0, 2, PERSON] = logit(SCORE_THR * 0.99)  # below the threshold: dropped

    d = postprocess(boxes, logits, (100, 200))

    np.testing.assert_allclose(d.boxes, [[80, 30, 120, 70]])
    np.testing.assert_allclose(d.scores, [0.9])


def test_postprocess_ranks_query_class_pairs_globally_before_the_threshold():
    # Two queries, two pairs above the threshold each; top-k keeps only the
    # query count (2) of pairs, so the weaker person pair is cut by two cars.
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]]])
    logits = np.full((1, 2, 91), -20.0)
    logits[0, 0, 3] = logit(0.9)
    logits[0, 1, 3] = logit(0.8)
    logits[0, 1, PERSON] = logit(0.5)

    d = postprocess(boxes, logits, (10, 10))

    assert len(d.scores) == 0


def test_postprocess_clips_to_the_image():
    logits = np.full((1, 1, 91), -20.0)
    logits[0, 0, PERSON] = logit(0.8)

    d = postprocess(np.array([[[0.0, 0.5, 0.4, 1.2]]]), logits, (60, 100))

    np.testing.assert_allclose(d.boxes, [[0, 0, 20, 60]])


class FakeSession:
    def __init__(self, size=32):
        self.size, self.feeds = size, []

    def get_inputs(self):
        return [type("I", (), {"name": "input", "shape": [1, 3, self.size, self.size]})]

    def run(self, names, feed):
        self.feeds.append(feed)
        logits = np.full((1, 1, 91), -20.0, np.float32)
        logits[0, 0, PERSON] = 3.0
        # Graph order differs from the order asked for: outputs are read by name.
        outputs = {"labels": logits, "dets": np.array([[[0.5, 0.5, 0.5, 0.5]]], np.float32)}
        return [outputs[n] for n in names or outputs]


def test_detect_maps_normalised_boxes_to_source_pixels():
    session = FakeSession()

    d = OnnxRfdetrDetector(session).detect(Frame(0, 0.0, np.zeros((100, 200, 3), np.uint8)))

    assert session.feeds[0]["input"].shape == (1, 3, 32, 32)
    np.testing.assert_allclose(d.boxes, [[50, 25, 150, 75]])


def test_the_exported_model_finds_people_in_a_real_frame():
    """Adapter against the exported RF-DETR Nano; needs the runtime group, the
    export (scripts/export-detector.py rfdetr-n) and scene 071's video."""
    import cv2

    from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import (
        NvidiaMtmcReader,
    )

    pytest.importorskip("onnxruntime", reason="runtime group not installed")
    data = Path(os.environ.get("FOOTFALL_DATA_DIR", "~/datasets/footfall-heatmaps"))
    data = data.expanduser()
    weights = data / "models" / "rf_detr_nano.onnx"
    video = NvidiaMtmcReader(data).video("071", "645")
    if not weights.exists() or not video.exists():
        pytest.skip("no RF-DETR Nano export or no scene 071 video")
    cap = cv2.VideoCapture(str(video))
    ok, image = cap.read()  # in memory only
    cap.release()
    detector = OnnxRfdetrDetector.load(weights)

    d = detector.detect(Frame(0, 0.0, image))
    flipped = detector.detect(Frame(0, 0.0, image[:, ::-1].copy()))

    assert detector.input_size == 384
    assert (d.scores > 0.5).sum() >= 3  # people in the first frame of scene 071
    # Mirrored frame: most strong people found again, mirrored back to the same
    # place (fails if the boxes are not scaled with the right axes).
    a = d.boxes[d.scores > 0.5]
    b = flipped.boxes[flipped.scores > 0.3][:, [2, 1, 0, 3]] * [-1, 1, -1, 1] + [1920, 0, 1920, 0]
    near = [np.abs(b - box).max(axis=1).min() for box in a]
    assert np.median(near) < 20
