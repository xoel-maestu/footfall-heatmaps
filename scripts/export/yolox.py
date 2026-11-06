"""Run YOLOX's official tools/export_onnx.py for YOLOX-s at another input size.

    python yolox.py <YOLOX checkout> <checkpoint> <output.onnx> <size>

Two shims, nothing else changed: torch.onnx._export (the private name the
exporter calls, gone from current torch) is torch.onnx.export with the
TorchScript exporter (dynamo=False), as when the release models were exported;
and the input size goes in through an experiment file (the official way: the
exporter's command-line opts cannot set a tuple). The checkpoint is loaded by
the exporter's torch.load, forced to weights_only=True (_weights_only.py).
"""

import functools
import runpy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))  # -I leaves the script's directory out
from _weights_only import force_weights_only  # noqa: E402

EXP = """
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33  # exps/default/yolox_s.py
        self.width = 0.50
        self.test_size = ({size}, {size})
        self.exp_name = "yolox_s_{size}"
"""


def main(src, ckpt, out, size):
    force_weights_only(torch)
    torch.onnx._export = functools.partial(torch.onnx.export, dynamo=False)
    sys.path.insert(0, src)
    with tempfile.TemporaryDirectory() as d:
        exp = Path(d) / f"yolox_s_{size}.py"
        exp.write_text(EXP.format(size=int(size)))
        sys.argv = ["export_onnx.py", "--output-name", out, "-f", str(exp), "-c", ckpt]
        runpy.run_path(str(Path(src) / "tools" / "export_onnx.py"), run_name="__main__")


if __name__ == "__main__":
    main(*sys.argv[1:])
