# Architecture

The package `space_usage_heatmaps` (under `src/`) is split into bounded
contexts, and most of them into three layers. Dependencies point inwards:
infrastructure depends on application and domain, application on domain,
and domain on nothing outside itself.

## Contexts

| Context | Layers | What it holds |
|---|---|---|
| `perception` | domain, application, infrastructure | frames, detections and tracks of one camera; the detectors (ONNX Runtime), trackers and video sources; privacy masks |
| `floorplan` | domain | the foot point of a box and the image-to-floor homography |
| `occupancy` | domain, application, infrastructure | the floor grid, k-suppression, 15-minute slots, multi-camera fusion and stitching; the SQLite slot store |
| `evaluation` | domain, application, infrastructure, `baseline` | metrics, scene splits and the bootstrap; dataset readers and the cached tracks; the optional Ultralytics reference recipe |

Outside the contexts, at the package root: `cli.py` (the `space-usage`
command), `web/` (the viewer's HTTP server and its static page),
`site_recording.py` and `e1_final.py`. They wire the contexts together and
may import any layer.

## Layer rules

`scripts/check-layers.sh` runs [import-linter](https://github.com/seddonym/import-linter)
with the contracts in `pyproject.toml` (`[tool.importlinter]`). `scripts/verify.sh`
runs it, so a violation fails CI.

1. **Domain imports only domain.** No `*.domain` package imports any
   `application` or `infrastructure` package (of any context), nor `cli` or
   `web`.
2. **Application imports only domain and application.** No `*.application`
   package imports any `infrastructure` package, nor `cli` or `web`.
   Use cases receive their adapters (detector, tracker, video source, slot
   store, dataset reader) through ports declared in domain as `Protocol`
   classes: `perception/domain/ports.py`, `SlotStore` in
   `occupancy/domain/slots.py`, `DatasetReader` and `TrackSource` in
   `evaluation/domain/`.
3. **Domain and application import no third-party package, except numpy and
   `ot` (POT).** Both are pure numerical libraries without I/O, and the grid,
   the homography and the metrics are array maths. Every other installed
   distribution is listed as forbidden by its import name, including the
   optional groups (`runtime`, `tensorrt`, `export`, `baseline`) and the
   planned model packages before they are installed. File, network, GPU and
   video access therefore stays in infrastructure.

`tests/test_layer_contracts.py` keeps rule 3 complete: it fails when an
installed distribution is missing from the forbidden list, so a new
dependency cannot slip into domain or application unnoticed.

## Further rules enforced by tests

- `evaluation/baseline` is the only place that imports ultralytics (AGPL-3.0),
  and nothing on the run path imports it
  (`tests/evaluation/test_baseline_isolation.py`).
- The perception adapters never import PyAV, and the lock never installs it
  (`tests/perception/infrastructure/test_dependencies.py`).
- The privacy rules R1 to R7 (see the README) each have a test under
  `tests/privacy/`.
