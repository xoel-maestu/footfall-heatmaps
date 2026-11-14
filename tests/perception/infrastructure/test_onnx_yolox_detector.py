import logging

import numpy as np
import pytest

from footfall_heatmaps.perception.domain.frame import Frame
from footfall_heatmaps.perception.infrastructure.onnx_yolox_detector import (
    CPU,
    CUDA,
    NMS_THR,
    SCORE_THR,
    OnnxYoloxDetector,
    decode,
    engine_of,
    nms,
    preprocess,
)


def test_preprocess_letterboxes_top_left_on_114_and_keeps_bgr():
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 0] = 10  # blue channel first, as OpenCV decodes

    x, ratio = preprocess(image, 64)

    assert ratio == pytest.approx(0.32)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    assert x[0, 0, 0] == 10 and x[1, 0, 0] == 0  # not normalised, not swapped to RGB
    assert (x[:, 32:, :] == 114).all()  # 100 * 0.32 = 32 rows of image, then padding


def test_decode_adds_the_grid_and_scales_by_the_stride():
    out = np.zeros((1, 8 * 8 + 4 * 4 + 2 * 2, 85), np.float32)
    # Cell (x=3, y=1) of the stride-8 grid: offset (0.5, 0.5), log size 0 -> 8 px.
    out[0, 1 * 8 + 3, :4] = [0.5, 0.5, 0.0, 0.0]
    out[0, 1 * 8 + 3, 4] = 0.8  # objectness
    out[0, 1 * 8 + 3, 5] = 0.5  # person

    boxes, scores = decode(out, 64)

    np.testing.assert_allclose(boxes[1 * 8 + 3], [24, 8, 32, 16])
    assert scores[1 * 8 + 3, 0] == pytest.approx(0.4)


def test_nms_suppresses_overlaps_above_the_threshold_keeping_the_best():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], float)

    assert nms(boxes, np.array([0.5, 0.9, 0.3]), 0.45) == [1, 2]


class FakeSession:
    """An ONNX session whose output puts one box at a given cell of the stride-8 grid."""

    def __init__(self, cells, size=64):
        self.cells, self.size, self.inputs = cells, size, []

    def get_inputs(self):
        return [type("I", (), {"name": "images", "shape": [1, 3, self.size, self.size]})]

    def run(self, _, feed):
        self.inputs.append(feed["images"])
        n = sum((self.size // s) ** 2 for s in (8, 16, 32))
        out = np.zeros((1, n, 85), np.float32)
        out[0, :, 4] = 1.0
        for cell, cls, score in self.cells:
            out[0, cell, :4] = [0.5, 0.5, np.log(2), np.log(2)]  # 16 px box
            out[0, cell, 5 + cls] = score
        return [out]


def frame(h=128, w=128):
    return Frame(0, 0.0, np.zeros((h, w, 3), np.uint8))


def test_detect_keeps_person_boxes_in_source_pixels():
    session = FakeSession([(1 * 8 + 3, 0, 0.9)])  # centre (28, 12) in the 64 px input

    d = OnnxYoloxDetector(session).detect(frame())  # 128 px source: ratio 0.5

    np.testing.assert_allclose(d.boxes, [[40, 8, 72, 40]])
    np.testing.assert_allclose(d.scores, [0.9])
    assert session.inputs[0].shape == (1, 3, 64, 64)


def test_detect_drops_other_classes_and_scores_at_or_below_the_threshold():
    session = FakeSession([(3, 2, 0.9), (20, 0, SCORE_THR), (40, 0, 0.5)])

    d = OnnxYoloxDetector(session).detect(frame())

    assert d.scores.tolist() == [0.5]


def test_nms_is_class_agnostic_as_the_official_demo():
    # A car (class 2) box over a weaker person box at the same cell suppresses it.
    session = FakeSession([(9, 2, 0.9), (9, 0, 0.5)])

    d = OnnxYoloxDetector(session).detect(frame())

    assert len(d.scores) == 0


def test_thresholds_are_the_official_onnx_demos():
    assert (SCORE_THR, NMS_THR) == (0.1, 0.45)


def test_cuda_is_asked_first_and_a_cpu_fallback_is_logged(caplog):
    with caplog.at_level(logging.WARNING):
        assert engine_of(["CPUExecutionProvider"]) == "CPUExecutionProvider"
    assert "CPU" in caplog.text

    caplog.clear()
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert engine_of(available) == "CUDAExecutionProvider"
    assert not caplog.text


def test_the_release_model_loads_and_detects_with_onnx_runtime():
    """Adapter against the real thing; needs the runtime group and the weights."""
    import os
    from pathlib import Path

    pytest.importorskip("onnxruntime", reason="runtime group not installed")
    data = Path(os.environ.get("FOOTFALL_DATA_DIR", "~/datasets/footfall-heatmaps"))
    path = data.expanduser() / "models" / "yolox_s.onnx"
    if not path.exists():
        pytest.skip(f"no YOLOX-s weights at {path}")

    detector = OnnxYoloxDetector.load(path)
    d = detector.detect(frame(1080, 1920))

    assert detector.input_size == 640 and detector.engine in (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert d.boxes.shape == (len(d.scores), 4)


ORT_OOM = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running "
    "Conv node. Name:'/backbone/Conv' Status Message: /onnxruntime_src/onnxruntime/core/"
    "framework/bfc_arena.cc:359 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, "
    "onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 52428800"
)


class OutOfMemorySession(FakeSession):
    def run(self, _, feed):
        raise RuntimeError(ORT_OOM)


def test_out_of_memory_on_cuda_retries_once_on_cpu_and_says_so(caplog):
    cpu = FakeSession([(1 * 8 + 3, 0, 0.9)])
    built = []
    detector = OnnxYoloxDetector(
        OutOfMemorySession([]), cpu_session=lambda: built.append(1) or cpu, engine=CUDA
    )

    with caplog.at_level(logging.WARNING):
        first = detector.detect(frame())
        second = detector.detect(frame())

    assert len(first.scores) == len(second.scores) == 1
    assert built == [1] and len(cpu.inputs) == 2
    assert detector.engine == CPU and detector.providers == [CUDA, CPU]
    assert len(caplog.records) == 1 and "CPU" in caplog.text and "memory" in caplog.text


def test_other_runtime_errors_are_not_retried():
    class Broken(FakeSession):
        def run(self, _, feed):
            raise RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : wrong input")

    detector = OnnxYoloxDetector(Broken([]), cpu_session=lambda: FakeSession([]), engine=CUDA)

    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        detector.detect(frame())


def test_out_of_memory_on_cpu_is_raised():
    detector = OnnxYoloxDetector(
        OutOfMemorySession([]), cpu_session=lambda: FakeSession([]), engine=CPU
    )

    with pytest.raises(RuntimeError, match="Failed to allocate"):
        detector.detect(frame())


def test_a_session_without_a_cpu_fallback_raises_out_of_memory():
    with pytest.raises(RuntimeError, match="Failed to allocate"):
        OnnxYoloxDetector(OutOfMemorySession([])).detect(frame())


def test_boxes_are_clipped_to_the_image():
    # Cell (0, 0): centre (4, 4), 16 px in the 64 px input -> (-8, -8, 24, 24) in source px;
    # on a 110 x 128 image (ratio 0.5) the last cell reaches past its bottom right.
    last = 8 * 8 - 1  # cell (7, 7): centre (60, 60) -> (104, 104, 136, 136)
    session = FakeSession([(0, 0, 0.9), (last, 0, 0.8)])

    d = OnnxYoloxDetector(session).detect(frame(h=110, w=128))

    np.testing.assert_allclose(d.boxes, [[0, 0, 24, 24], [104, 104, 128, 110]])


def test_the_release_model_is_invariant_to_grey_padding_below_a_real_frame():
    """Padding at the bottom with the letterbox's own 114 grey, by a multiple of
    the resize step (1080p: 3 rows per input row), leaves the letterboxed input
    unchanged where the image is: the same people, boxes and scores. Fails if
    the boxes are not mapped back with the ratio, or the padding is not 114 at
    the bottom. Needs the runtime group, the weights and scene 071's video."""
    import os
    from pathlib import Path

    import cv2

    from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import (
        NvidiaMtmcReader,
    )

    pytest.importorskip("onnxruntime", reason="runtime group not installed")
    data = Path(os.environ.get("FOOTFALL_DATA_DIR", "~/datasets/footfall-heatmaps"))
    data = data.expanduser()
    weights = data / "models" / "yolox_s.onnx"
    video = NvidiaMtmcReader(data).video("071", "645")
    if not weights.exists() or not video.exists():
        pytest.skip("no YOLOX-s weights or no scene 071 video")
    cap = cv2.VideoCapture(str(video))
    ok, image = cap.read()  # in memory only
    cap.release()
    assert ok and image.shape == (1080, 1920, 3)
    padded = np.full((1080 + 210, 1920, 3), 114, np.uint8)
    padded[:1080] = image
    detector = OnnxYoloxDetector.load(weights)

    a = detector.detect(Frame(0, 0.0, image))
    b = detector.detect(Frame(0, 0.0, padded))

    assert (a.scores > 0.5).sum() >= 3  # people in the first frame of scene 071
    order_a, order_b = np.lexsort(a.boxes.T[::-1]), np.lexsort(b.boxes.T[::-1])
    np.testing.assert_allclose(b.boxes[order_b], a.boxes[order_a], atol=0.5)
    np.testing.assert_allclose(b.scores[order_b], a.scores[order_a], atol=1e-3)


def test_score_and_nms_thresholds_are_parameters():
    # Two person boxes at neighbouring cells: 16 px wide, 8 px apart, IoU 1/3.
    session = FakeSession([(9, 0, 0.9), (10, 0, 0.5), (40, 0, 0.3)])

    default = OnnxYoloxDetector(session).detect(frame())
    strict = OnnxYoloxDetector(session, score_thr=0.4).detect(frame())
    loose_nms = OnnxYoloxDetector(session, nms_thr=0.3).detect(frame())

    np.testing.assert_allclose(default.scores, [0.9, 0.5, 0.3], rtol=1e-6)
    np.testing.assert_allclose(strict.scores, [0.9, 0.5], rtol=1e-6)
    np.testing.assert_allclose(loose_nms.scores, [0.9, 0.3], rtol=1e-6)  # IoU .36 > 0.3


def test_the_recipe_names_the_thresholds_it_ran_with():
    from footfall_heatmaps.perception.infrastructure.onnx_yolox_detector import recipe

    assert (recipe()["score_thr"], recipe()["nms_thr"]) == (0.1, 0.45)
    assert recipe(score_thr=0.05, nms_thr=0.6)["score_thr"] == 0.05
    assert recipe(score_thr=0.05, nms_thr=0.6)["nms_thr"] == 0.6
