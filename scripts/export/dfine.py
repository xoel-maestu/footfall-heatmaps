"""Run D-FINE's official tools/deployment/export_onnx.py on a release checkpoint.

    python dfine.py <D-FINE checkout> <checkpoint> <output.onnx> <config yml>

Three shims, nothing else changed: torch.onnx.export runs the TorchScript
exporter (dynamo=False), the default when the exporter was written; current
torch defaults to the dynamo one. And calflops, which src/misc imports for a
FLOPs profiler the export never calls, is an empty module: the real one
imports transformers at import time. And the exporter's dummy input is
one image, not 32: tracing 32 images on CPU takes more than 14 GB of RAM, and
the pipeline runs one image per call (traced at batch 1, the decoder's
`if memory.shape[0] > 1` takes the batch-1 branch). The checkpoint is loaded by the exporter's
torch.load, weights_only=True by default in torch >= 2.6. The exporter writes
<checkpoint>.onnx next to the checkpoint; it is moved to <output.onnx>.
"""

import functools
import runpy
import shutil
import sys
import types
from pathlib import Path

import torch


def main(src, ckpt, out, config):
    torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
    sys.modules["calflops"] = types.SimpleNamespace(calculate_flops=None)
    rand = torch.rand
    torch.rand = lambda n, *shape, **kw: rand(1 if n == 32 else n, *shape, **kw)
    sys.argv = ["export_onnx.py", "-c", config, "-r", ckpt]
    runpy.run_path(str(Path(src) / "tools" / "deployment" / "export_onnx.py"), run_name="__main__")
    shutil.move(ckpt.replace(".pth", ".onnx"), out)


if __name__ == "__main__":
    main(*sys.argv[1:])
