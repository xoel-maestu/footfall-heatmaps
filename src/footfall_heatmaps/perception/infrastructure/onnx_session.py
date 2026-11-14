"""ONNX Runtime sessions for the detector adapters: TensorRT (fp16), CUDA or
CPU, each failure logged and handed to the next engine, at load and when the
GPU runs out of memory during a run.

Engines (the config's engine):
- onnx-cuda: CUDA execution provider, fp32; CPU if CUDA is missing.
- onnx-trt: TensorRT execution provider, fp16, its engines cached on disk next
  to the model (models/tensorrt/); nodes TensorRT cannot run go to CUDA. A
  missing TensorRT (the optional tensorrt group), a failed engine build or an
  out-of-memory error falls back to CUDA, then to CPU.

Only the model file goes through ONNX Runtime; nothing else in it is run.
"""

import ctypes
import logging
import re
from pathlib import Path

import numpy as np

TENSORRT, CUDA, CPU = "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
# What ONNX Runtime's messages say when the GPU runs out of memory during run:
# the BFC arena, a CUDA allocation, cuDNN or cuBLAS workspaces.
OUT_OF_MEMORY = re.compile(r"Failed to allocate memory|out of memory|_ALLOC_FAILED")
# The TensorRT 10 libraries of the tensorrt-cu13-libs wheel that ONNX Runtime's
# TensorRT provider links against, in dependency order.
TENSORRT_LIBS = ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10")
TENSORRT_MISSING = "TensorRT is not installed: uv sync --group runtime --group tensorrt"
# Builder workspace: a quarter of the 4 GB card, so a build leaves room for the rest.
TENSORRT_WORKSPACE = 1 << 30

log = logging.getLogger(__name__)


def engine_of(providers):
    """The provider a session runs on; CPU is logged as a fallback."""
    if providers[0] == CPU:
        log.warning("ONNX Runtime is running the detector on %s, not the GPU: slow", providers[0])
    return providers[0]


def clip_to_image(boxes, shape):
    """(N, 4) x1 y1 x2 y2 boxes clipped to an image of shape (h, w, ...)."""
    h, w = shape[:2]
    return np.clip(np.reshape(boxes, (-1, 4)), 0, [w, h, w, h])


def open_session(chain):
    """chain: [(provider, factory)] in order of preference. The first factory
    whose session builds and runs on its provider -> (session, provider, the
    rest of the chain). A failure is logged and the next one tried; the last
    one's session is taken as it comes, and its error raised."""
    for k, (provider, factory) in enumerate(chain):
        last = k == len(chain) - 1
        try:
            session = factory()
        except Exception as e:  # onnxruntime's exceptions are not RuntimeErrors
            if last:
                raise
            log.warning(
                "ONNX Runtime could not start %s (%s); trying %s", provider, e, chain[k + 1][0]
            )
            continue
        actual = provider if last else session.get_providers()[0]
        if actual == provider:
            return session, actual, chain[k + 1 :]
        log.warning(
            "ONNX Runtime has no %s (it chose %s); trying %s", provider, actual, chain[k + 1][0]
        )
    raise ValueError("no engine to open")


def tensorrt_options(cache_dir):
    """TensorRT provider options: fp16, engines and timing cached in cache_dir."""
    return {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir),
        "trt_max_workspace_size": TENSORRT_WORKSPACE,
    }


def _preload_tensorrt():
    """Load the wheel's TensorRT libraries so ONNX Runtime's provider finds them."""
    try:
        import tensorrt_libs
    except ImportError as e:
        raise OSError(TENSORRT_MISSING) from e
    directory = Path(tensorrt_libs.__file__).parent
    for name in TENSORRT_LIBS:
        ctypes.CDLL(str(directory / name), mode=ctypes.RTLD_GLOBAL)


def engine_chain(onnxruntime, path, engine):
    """[(provider, session factory)] of an engine, best first."""

    def session(*providers):
        return lambda: onnxruntime.InferenceSession(str(path), providers=list(providers))

    chain = [(CUDA, session(CUDA, CPU)), (CPU, session(CPU))]
    if engine == "onnx-trt":
        cache = Path(path).parent / "tensorrt"

        def tensorrt():
            _preload_tensorrt()
            cache.mkdir(parents=True, exist_ok=True)
            return session((TENSORRT, tensorrt_options(cache)), CUDA, CPU)()

        chain.insert(0, (TENSORRT, tensorrt))
    elif engine != "onnx-cuda":
        raise ValueError(f"no ONNX Runtime engine {engine!r}: onnx-cuda or onnx-trt")
    return chain


class OnnxDetector:
    def __init__(self, session, cpu_session=None, engine=None, fallbacks=None):
        """session: an onnxruntime.InferenceSession of the model, or a test double.
        fallbacks: [(provider, session factory)] tried in order if the GPU runs
        out of memory during a run; cpu_session() alone is [(CPU, it)].
        engine: session's provider. providers: every provider this detector has
        run on, in order."""
        self._session = session
        if fallbacks is None:
            fallbacks = [] if cpu_session is None else [(CPU, cpu_session)]
        self._fallbacks = list(fallbacks)
        self.engine = engine
        self.providers = [engine] if engine else []
        inp = session.get_inputs()[0]
        self._input, self.input_size = inp.name, int(inp.shape[-1])

    @classmethod
    def load(cls, path, engine="onnx-cuda", **params):
        """The model on the engine's best provider that starts (onnx_session docstring)."""
        import onnxruntime

        onnxruntime.preload_dlls()  # the CUDA and cuDNN libraries of the nvidia-* wheels
        onnxruntime.disable_telemetry_events()  # privacy R1: no usage events leave the process
        session, provider, rest = open_session(engine_chain(onnxruntime, path, engine))
        return cls(session, engine=engine_of([provider]), fallbacks=rest, **params)

    def _run(self, feed, outputs=None):
        """The model's outputs named in outputs (all, in graph order, if None)
        for feed (input name -> array)."""
        while True:
            try:
                return self._session.run(outputs, feed)
            except Exception as e:  # onnxruntime's RuntimeException is not a RuntimeError
                if self.engine == CPU or not self._fallbacks or not OUT_OF_MEMORY.search(str(e)):
                    raise
            log.warning(
                "ONNX Runtime ran out of GPU memory on %s; retrying on %s",
                self.engine,
                self._fallbacks[0][0],
            )
            self._session, self.engine, self._fallbacks = open_session(self._fallbacks)
            self.providers.append(self.engine)
