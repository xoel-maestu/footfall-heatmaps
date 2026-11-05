"""Export the RF-DETR Nano COCO checkpoint to ONNX with the rfdetr package's own exporter.

    python rfdetr.py <work dir> <checkpoint> <output.onnx>

Runs in scripts/export/rfdetr's own environment. The checkpoint is loaded by
rfdetr's loader with torch.load forced to weights_only=True: a load that asks
for full unpickling raises instead. Hugging Face is offline: nothing else is
downloaded. The rest is RFDETRNano(pretrain_weights=...).export(), defaults
(opset 17, batch 1, the model's 384 x 384).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402

_load = torch.load


def _weights_only(*args, **kwargs):
    if kwargs.get("weights_only") is False:
        raise RuntimeError("refusing to unpickle a checkpoint (weights_only=False)")
    return _load(*args, **(kwargs | {"weights_only": True}))


def main(_work, ckpt, out):
    torch.load = _weights_only
    from rfdetr import RFDETRNano

    with tempfile.TemporaryDirectory() as d:
        weights = Path(d) / "rf-detr-nano.pth"  # the name rfdetr knows the Nano weights by
        shutil.copyfile(ckpt, weights)
        model = RFDETRNano(pretrain_weights=str(weights), device="cpu")
        path = model.export(output_dir=str(Path(d) / "out"), format="onnx")
        shutil.move(str(path), out)


if __name__ == "__main__":
    main(*sys.argv[1:])
