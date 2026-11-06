"""The engine chain of the ONNX detectors: TensorRT, then CUDA, then CPU, each
step logged, at load and when the GPU runs out of memory during a run."""

import logging

import numpy as np
import pytest

from space_usage_heatmaps.perception.domain.frame import Frame
from space_usage_heatmaps.perception.infrastructure.onnx_session import (
    CPU,
    CUDA,
    TENSORRT,
    open_session,
    tensorrt_options,
)
from space_usage_heatmaps.perception.infrastructure.onnx_yolox_detector import OnnxYoloxDetector
from tests.perception.infrastructure.test_onnx_yolox_detector import FakeSession, OutOfMemorySession


def on(provider, session):
    session.get_providers = lambda: [provider, CPU]
    return session


def frame():
    return Frame(0, 0.0, np.zeros((128, 128, 3), np.uint8))


def test_the_first_engine_that_builds_is_used_and_the_rest_stay_as_fallbacks(caplog):
    trt = on(TENSORRT, FakeSession([]))
    chain = [(TENSORRT, lambda: trt), (CUDA, lambda: 1 / 0), (CPU, lambda: 1 / 0)]

    with caplog.at_level(logging.WARNING):
        session, provider, rest = open_session(chain)

    assert (session, provider) == (trt, TENSORRT)
    assert [p for p, _ in rest] == [CUDA, CPU]
    assert not caplog.text


def test_a_tensorrt_build_failure_falls_back_to_cuda_with_a_warning(caplog):
    def broken():
        raise RuntimeError("TensorRT EP failed to create engine from network")

    cuda = on(CUDA, FakeSession([]))
    chain = [(TENSORRT, broken), (CUDA, lambda: cuda), (CPU, lambda: on(CPU, FakeSession([])))]

    with caplog.at_level(logging.WARNING):
        session, provider, rest = open_session(chain)

    assert (session, provider, [p for p, _ in rest]) == (cuda, CUDA, [CPU])
    assert "TensorrtExecutionProvider" in caplog.text and "CUDAExecutionProvider" in caplog.text


def test_an_engine_onnx_runtime_silently_skipped_counts_as_a_failure(caplog):
    # Without the TensorRT libraries ONNX Runtime builds the session on the next provider.
    chain = [
        (TENSORRT, lambda: on(CUDA, FakeSession([]))),
        (CUDA, lambda: on(CUDA, FakeSession([]))),
    ]

    with caplog.at_level(logging.WARNING):
        _, provider, _ = open_session(chain)

    assert provider == CUDA and "TensorrtExecutionProvider" in caplog.text


def test_the_last_engine_failing_raises():
    with pytest.raises(ZeroDivisionError):
        open_session([(CUDA, lambda: 1 / 0), (CPU, lambda: 1 / 0)])


def test_out_of_memory_on_tensorrt_goes_to_cuda_then_cpu(caplog):
    cpu = on(CPU, FakeSession([(1 * 8 + 3, 0, 0.9)]))
    fallbacks = [(CUDA, lambda: on(CUDA, OutOfMemorySession([]))), (CPU, lambda: cpu)]
    detector = OnnxYoloxDetector(OutOfMemorySession([]), engine=TENSORRT, fallbacks=fallbacks)

    with caplog.at_level(logging.WARNING):
        d = detector.detect(frame())

    assert len(d.scores) == 1
    assert detector.engine == CPU and detector.providers == [TENSORRT, CUDA, CPU]
    assert caplog.text.count("memory") == 2


def test_tensorrt_runs_fp16_with_its_engines_cached_on_disk(tmp_path):
    options = tensorrt_options(tmp_path)

    assert options["trt_fp16_enable"] is True
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_engine_cache_path"] == str(tmp_path)
