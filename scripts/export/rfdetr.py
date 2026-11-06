"""Export the RF-DETR Nano COCO checkpoint to ONNX with the rfdetr package's own exporter.

    python rfdetr.py <checkpoint> <output.onnx>

Runs in scripts/export/rfdetr's own environment. The checkpoint is loaded by
rfdetr's loader with torch.load forced to weights_only=True (_weights_only.py).
Hugging Face is offline: nothing else is downloaded. The rest is
RFDETRNano(pretrain_weights=...).export(), defaults (opset 17, batch 1, the
model's 384 x 384).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402

sys.path.append(str(Path(__file__).parent))  # -I leaves the script's directory out
from _weights_only import force_weights_only  # noqa: E402


def main(ckpt, out):
    force_weights_only(torch)
    from rfdetr import RFDETRNano

    with tempfile.TemporaryDirectory() as d:
        weights = Path(d) / "rf-detr-nano.pth"  # the name rfdetr knows the Nano weights by
        shutil.copyfile(ckpt, weights)
        model = RFDETRNano(pretrain_weights=str(weights), device="cpu")
        path = model.export(output_dir=str(Path(d) / "out"), format="onnx")
        shutil.move(str(path), out)


if __name__ == "__main__":
    main(*sys.argv[1:])
