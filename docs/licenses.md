# Licences

> Every dependency, model weight and dataset this repo uses, with its licence
> and whether it is on the run path or for evaluation only. Read it before
> adding a dependency, a model or a dataset: the run path takes Apache-2.0,
> MIT or BSD code and weights only (spec 3).

Versions are the ones in `uv.lock`; licences are what each distribution
declares in its package metadata.

## Run path: what `space-usage` needs to produce a heatmap

Default dependencies (`uv sync`) and the `runtime` group (`uv sync --group runtime`).

| Distribution | Version | Licence | Why it is here |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | arrays everywhere |
| pot | 0.9.7.post1 | MIT | earth mover's distance of the metrics |
| pyyaml | 6.0.3 | MIT | `configs/*.yaml` |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (bundles FFmpeg, LGPL-2.1) | video decoding, letterbox resize |
| trackers | 2.6.1 | Apache-2.0 | ByteTrack (`ByteTrackTracker`) and OC-SORT (`OCSORTTracker`); its port of TrackEval's HOTA and IDF1 (TrackEval is MIT) for the tracking guards of `space-usage eval` |
| scipy | 1.18.1 | BSD-3-Clause | Hungarian assignment of the floor-plane tracker (`ucmc_tracker.py`); also pulled by pot and trackers |
| onnxruntime-gpu | 1.31.1 | MIT | runs the YOLOX, D-FINE and RF-DETR ONNX models (`runtime` group) |
| nvidia-ml-py | 13.615.71 | BSD | VRAM in `space-usage bench` (`runtime` group) |
| pillow | 13.0.0 | MIT-CMU | D-FINE's letterbox resize, its official bilinear filter (`runtime` group; also pulled by matplotlib) |

Transitive, pulled by the ones above:

| Distribution | Licence | Pulled by |
|---|---|---|
| supervision | MIT | trackers (its `Detections` type) |
| matplotlib, contourpy, cycler, kiwisolver, fonttools, pyparsing, python-dateutil | PSF-based (matplotlib), BSD, MIT | supervision; not called here |
| requests, urllib3, idna, charset-normalizer, certifi | Apache-2.0, MIT, BSD-3-Clause, MIT, MPL-2.0 | supervision, trackers; not called here |
| tqdm | MPL-2.0 AND MIT | supervision |
| defusedxml | PSF | supervision |
| pydeprecate | Apache-2.0 | supervision, trackers |
| rich, markdown-it-py, mdurl, pygments | MIT, MIT, MIT, BSD-2-Clause | trackers |
| packaging | Apache-2.0 OR BSD-2-Clause | onnxruntime-gpu |
| protobuf | BSD-3-Clause | onnxruntime-gpu |
| flatbuffers | Apache-2.0 | onnxruntime-gpu |
| tzdata | Apache-2.0 | matplotlib |
| nvidia-cuda-runtime, nvidia-cuda-nvrtc, nvidia-cublas, nvidia-cufft, nvidia-curand, nvidia-nvjitlink, nvidia-cudnn-cu13 | NVIDIA proprietary (CUDA and cuDNN EULAs, redistributable) | onnxruntime-gpu `[cuda,cudnn]` |

Notes:

- MPL-2.0 (certifi, tqdm) is file-level copyleft: it applies to changes of
  those files, which are used unmodified.
- The FFmpeg inside the OpenCV wheel is LGPL-2.1, dynamically linked by that
  wheel and used unmodified.
- The CUDA and cuDNN libraries are NVIDIA's runtime for the GPU, like the
  driver: not Apache, MIT or BSD, and not replaceable on an NVIDIA card. The
  CPU fallback of ONNX Runtime runs without them.
- `opencv-python` (the GUI build that trackers and ultralytics require) is
  overridden in `pyproject.toml` and never installed.
- `av` (PyAV), which supervision requires, is overridden the same way and
  never installed. PyAV itself is BSD-3-Clause, but its wheels bundle an
  FFmpeg (libavcodec and the rest) under LGPL-3.0+ that links libx264 and
  libx265, both GPL-2.0+. supervision imports it only inside its video
  helpers, which nothing here calls;
  `tests/perception/infrastructure/test_dependencies.py` checks that the
  perception adapters never import it and that the lock never installs it.

Optional engine: the `tensorrt` group (`uv sync --group runtime --group
tensorrt`) for `engine: onnx-trt` (ONNX Runtime's TensorRT execution provider,
fp16; `perception/infrastructure/onnx_session.py`).

| Distribution | Version | Licence | Why it is here |
|---|---|---|---|
| tensorrt-cu13-libs | 10.16.1.11 | NVIDIA proprietary (TensorRT Software License Agreement) | `libnvinfer.so.10` and the ONNX parser that ONNX Runtime's TensorRT provider loads; from NVIDIA's index `pypi.nvidia.com` |

- TensorRT is not open source and not on the Apache, MIT or BSD list of the
  run path. It is an optional speed-up only: without the group, `onnx-trt`
  logs a warning and runs on CUDA (then CPU), and `onnx-cuda` never loads it.
- The engines TensorRT builds (`SPACE_USAGE_DATA_DIR/models/tensorrt/`) are
  derived from the Apache-2.0 weights for this card and driver; they stay
  under the data dir, never committed.

Code written here after a published method:

| Module | After | Licence of the original | How |
|---|---|---|---|
| `perception/infrastructure/ucmc_tracker.py` | UCMCTrack, github.com/corfyi/UCMCTrack at `7851a49` (Copyright (c) 2023 Kefu Yi) | MIT | own implementation of its tracker and floor mapping, read but not copied; its filterpy and lap dependencies replaced by numpy and scipy |

## Model weights

Downloaded or exported into `SPACE_USAGE_DATA_DIR/models`, never committed;
loaded only by ONNX Runtime (run path) or by ultralytics (evaluation). Every
run-path file's sha256 is checked on every load
(`perception/infrastructure/model_weights.py`). Exported files come from the
release checkpoint below through the family's official exporter
(`scripts/export-detector.py <name>`, CPU, deterministic: a re-export gives the
same sha256).

| Weights | Source | sha256 | Licence | Use |
|---|---|---|---|---|
| `yolox_s.onnx` (YOLOX-s, COCO, 640) | Megvii-BaseDetection/YOLOX GitHub release `0.1.1rc0` | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` | Apache-2.0 | run path, `configs/h1-yolox-s.yaml` |
| `yolox_s_960.onnx` (YOLOX-s, COCO, 960) | exported from `yolox_s.pth` (same release, sha256 `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30`) by YOLOX 0.3.0's `tools/export_onnx.py` | `95316ad4705db4dff9d77c22238514d685c86c3cb6a7b9ceb46879bedc84e73b` | Apache-2.0 | run path, `configs/h2-yolox-s-960.yaml` |
| `yolox_s_1280.onnx` (YOLOX-s, COCO, 1280) | exported the same way at 1280 | `b89fb6acc9072a74ece25ab4951b43fd21328c2eb07e4d7606220c0aa97fa33d` | Apache-2.0 | candidate, `configs/h2-yolox-s-1280.yaml` |
| `dfine_s_obj2coco.onnx` (D-FINE-S, Objects365 then COCO, 640) | exported from `dfine_s_obj2coco.pth` (Peterande/storage GitHub release `dfinev1.0`, sha256 `75bedc333cdf665f9cddd2c6a2a9baa34e0e5e9a38db7bb8e6110edc3b4ba8eb`) by Peterande/D-FINE `tools/deployment/export_onnx.py` at `956d170` | `f29f9569ef1d089dce6b392aef886fe4afd15e6193838bedae6e2ee4830b6ddd` | Apache-2.0 (the D-FINE repository's licence, which its release weights carry) | run path candidate, `configs/h1-dfine-s.yaml` |
| `rf_detr_nano.onnx` (RF-DETR Nano, COCO, 384) | exported from `rf-detr-nano.pth` (Roboflow's public bucket `storage.googleapis.com/rfdetr/nano_coco/`, sha256 `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`, md5 as recorded by rfdetr 1.11.2) by `RFDETRNano.export()` of rfdetr 1.11.2 | `d217512443ea74ca6f025e9d9239cfd7650de318664c6fc26ee81ae705e4baac` | Apache-2.0 (rfdetr's README licence table: N, S, M, L Apache-2.0; XL and 2XL PML 1.0, not used) | run path candidate, `configs/h1-rfdetr-n.yaml` |
| `yolov8n.pt` (YOLOv8n, COCO) | ultralytics/assets GitHub release, fetched by ultralytics | not pinned | AGPL-3.0 | evaluation only: the Ultralytics reference recipe |

Training data of these weights: COCO (all), and Objects365 for D-FINE-S's
pre-training. The weights' own licence is the vendor's Apache-2.0.

## Export only

Turn release checkpoints into the ONNX files above; never imported by the
package and never installed by CI.

The optional `export` group (`uv sync --group runtime --group export`) runs the
YOLOX and D-FINE exporters (`scripts/export/yolox.py`, `scripts/export/dfine.py`):

| Distribution | Version | Licence | Why |
|---|---|---|---|
| torch, torchvision | 2.14.1+cu130, 0.29.1+cu130 | BSD-3-Clause with Apache-2.0, BSL-1.0 and MIT parts; BSD | the exporters trace the models |
| onnx, onnxsim | 1.23.2, 0.8.1 | Apache-2.0; MIT AND (Apache-2.0 OR BSD-2-Clause) | saving and simplifying the graph |
| loguru, thop, tabulate | 0.7.3, 0.1.1.post2209072238, 0.10.0 | MIT | imported by the yolox package |
| faster-coco-eval, tensorboard | 1.8.0, 2.21.0 | Apache-2.0 | imported by D-FINE's package (`calflops` is stubbed, not installed: it pulls transformers) |
| absl-py, grpcio, markdown, werkzeug, tensorboard-data-server, ml-dtypes | transitive | Apache-2.0, Apache-2.0, BSD-3-Clause, BSD-3-Clause, Apache-2.0, Apache-2.0 | tensorboard, onnx |

RF-DETR's exporter has its own locked environment, `scripts/export/rfdetr`
(rfdetr pins pyDeprecate < 0.10, which would downgrade the run path's
trackers dependency): rfdetr 1.11.2 (Apache-2.0) with transformers 5.22.1,
huggingface-hub, tokenizers, safetensors (Apache-2.0), pydantic and
supervision (MIT), torch and torchvision as above. Its checkpoint download is
the only weights download from outside GitHub (`EXPORT_HOSTS`); the run path's
allowlist stays GitHub's release hosts. Installing that environment (`uv run
--project scripts/export/rfdetr`) fetches packages from PyPI and torch and
torchvision from `download.pytorch.org`, as the `export` group does.

## Evaluation only

The optional `baseline` group (`uv sync --group baseline`) reproduces the
Ultralytics reference recipe. It is imported only under
`src/space_usage_heatmaps/evaluation/baseline/`, never by the run path
(`tests/evaluation/test_baseline_isolation.py`).

| Distribution | Version | Licence |
|---|---|---|
| ultralytics | 8.4.202 | AGPL-3.0 |
| torch, torchvision | 2.14.1+cu130, 0.29.1+cu130 | BSD-3-Clause with Apache-2.0, BSL-1.0 and MIT parts (torch); BSD (torchvision) |
| lap | 0.5.14 | BSD-2-Clause |
| polars, psutil | 2.1.0, 7.2.2 | MIT, BSD-3-Clause |

## Datasets

Kept under `SPACE_USAGE_DATA_DIR`, never committed, used to measure only.

| Dataset | Licence | Consequence |
|---|---|---|
| NVIDIA PhysicalAI-SmartSpaces, MTMC_Tracking_2024 | Conflict: CC BY 4.0 on the Hugging Face card; the AI City Challenge 2024 data agreement forbids commercial use | measurement only; a product would replace it with its own footage |
| MOT17 (planned: MOT17-02, 04, 09) | CC BY-NC-SA 3.0 | non-commercial measurement only |

## Anti-patterns

- A dependency added to the default group or to `runtime` without a row here
  and a licence on the run-path list.
- Weights on the run path without a pinned sha256, or with a non-commercial or
  AGPL licence.
- Importing ultralytics, or anything under `evaluation/baseline`, from the
  pipeline.
- Training or tuning on a non-commercial dataset and shipping the result.
