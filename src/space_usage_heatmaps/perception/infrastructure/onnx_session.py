"""An ONNX Runtime session on CUDA with a logged CPU fallback, shared by the detector adapters.

Only the model file goes through ONNX Runtime; nothing else in it is run.
"""

import logging
import re

CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
# What ONNX Runtime's messages say when the GPU runs out of memory during run:
# the BFC arena, a CUDA allocation, cuDNN or cuBLAS workspaces.
OUT_OF_MEMORY = re.compile(r"Failed to allocate memory|out of memory|_ALLOC_FAILED")

log = logging.getLogger(__name__)


def engine_of(providers):
    """The provider a session runs on; anything but CUDA is logged as a fallback."""
    if providers[0] != CUDA:
        log.warning("ONNX Runtime is running the detector on %s, not CUDA: slow", providers[0])
    return providers[0]


class OnnxDetector:
    def __init__(self, session, cpu_session=None, engine=None):
        """session: an onnxruntime.InferenceSession of the model, or a test double.
        cpu_session(): the same model on CPUExecutionProvider, built only if the
        GPU runs out of memory during a run (spec 4). engine: session's provider.
        providers: every provider this detector has run on, in order."""
        self._session, self._cpu_session = session, cpu_session
        self.engine = engine
        self.providers = [engine] if engine else []
        inp = session.get_inputs()[0]
        self._input, self.input_size = inp.name, int(inp.shape[-1])

    @classmethod
    def load(cls, path):
        """ONNX Runtime on CUDA, falling back to CPU (logged), at load or on out of memory."""
        import onnxruntime

        onnxruntime.preload_dlls()  # the CUDA and cuDNN libraries of the nvidia-* wheels
        onnxruntime.disable_telemetry_events()  # privacy R1: no usage events leave the process
        session = onnxruntime.InferenceSession(str(path), providers=[CUDA, CPU])
        return cls(
            session,
            cpu_session=lambda: onnxruntime.InferenceSession(str(path), providers=[CPU]),
            engine=engine_of(session.get_providers()),
        )

    def _run(self, feed):
        """Every output of the model for feed (input name -> array)."""
        try:
            return self._session.run(None, feed)
        except Exception as e:  # onnxruntime's RuntimeException is not a RuntimeError
            if self.engine == CPU or self._cpu_session is None or not OUT_OF_MEMORY.search(str(e)):
                raise
        log.warning("ONNX Runtime ran out of GPU memory on %s; retrying on CPU: slow", self.engine)
        self._session, self.engine = self._cpu_session(), CPU
        self.providers.append(CPU)
        return self._session.run(None, feed)
