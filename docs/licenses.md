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
| trackers | 2.6.1 | Apache-2.0 | ByteTrack (`ByteTrackTracker`) |
| onnxruntime-gpu | 1.31.1 | MIT | runs the YOLOX ONNX model (`runtime` group) |
| nvidia-ml-py | 13.615.71 | BSD | VRAM in `space-usage bench` (`runtime` group) |

Transitive, pulled by the ones above:

| Distribution | Licence | Pulled by |
|---|---|---|
| supervision | MIT | trackers (its `Detections` type) |
| scipy | BSD-3-Clause | pot, trackers (Hungarian assignment) |
| av | BSD-3-Clause (bundles FFmpeg, LGPL-2.1) | supervision; not called here |
| matplotlib, contourpy, cycler, kiwisolver, fonttools, pyparsing, python-dateutil, pillow | PSF-based (matplotlib), BSD, MIT, MIT-CMU (pillow) | supervision; not called here |
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
- The FFmpeg inside the OpenCV and PyAV wheels is LGPL-2.1, dynamically
  linked by those wheels and used unmodified.
- The CUDA and cuDNN libraries are NVIDIA's runtime for the GPU, like the
  driver: not Apache, MIT or BSD, and not replaceable on an NVIDIA card. The
  CPU fallback of ONNX Runtime runs without them.
- `opencv-python` (the GUI build that trackers and ultralytics require) is
  overridden in `pyproject.toml` and never installed.

## Model weights

Downloaded into `SPACE_USAGE_DATA_DIR/models`, never committed; loaded only by
ONNX Runtime (run path) or by ultralytics (evaluation).

| Weights | Source | sha256 | Licence | Use |
|---|---|---|---|---|
| `yolox_s.onnx` (YOLOX-s, COCO) | Megvii-BaseDetection/YOLOX GitHub release `0.1.1rc0` | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, checked on every load (`perception/infrastructure/model_weights.py`) | Apache-2.0 | run path, `configs/h1-yolox-s.yaml` |
| `yolov8n.pt` (YOLOv8n, COCO) | ultralytics/assets GitHub release, fetched by ultralytics | not pinned | AGPL-3.0 | evaluation only: the Ultralytics reference recipe |

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
