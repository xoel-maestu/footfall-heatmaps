"""space-usage: command-line entry point (driving adapter)."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from space_usage_heatmaps.evaluation.application.evaluate_oracle import (
    SOURCES,
    TRUTHS,
    evaluate_oracle,
)
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader

LAYERS = ["dwell", "passage"]
METRICS = ["mass_error", "tvd", "wape_zone", "emd_m", "pearson"]
TRUTH_TITLES = {
    "visible@coverage": "people this camera sees, on the coverage mask (E1 primary)",
    "full@coverage": "every person, on the coverage mask (E2a target)",
    "visible@visibility": (
        "people this camera sees, on cells where it sees someone (diagnostic, "
        "hides false positives)"
    ),
}


def data_dir():
    return Path(
        os.environ.get("SPACE_USAGE_DATA_DIR", "~/datasets/space-usage-heatmaps")
    ).expanduser()


def _json_value(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _cell(v, width=12, missing="n/a"):
    return f"{missing:>{width}}" if v is None else f"{v:>{width}.4f}"


# Table text for an undefined WAPE, from LayerMetrics.wape_zone_reason.
WAPE_REASONS = {"computed for dwell only": "dwell only"}


def _metric(m, name):
    """One metric of a LayerMetrics as a table cell; an undefined WAPE shows why."""
    missing = WAPE_REASONS.get(m.wape_zone_reason, "no 1% zone") if name == "wape_zone" else "n/a"
    return _cell(getattr(m, name), missing=missing)


def _write_report(r):
    report = {
        "dataset": r.dataset,
        "scene": r.scene,
        "camera": r.camera,
        "cell_m": r.cell_m,
        "source": r.source,
        "occlusion_fraction": r.occlusion_fraction,
        "coverage_cells": r.coverage_cells,
        "visibility_cells": r.visibility_cells,
        "samples": r.samples,
        "outside_grid": r.dropped,
        "results": {
            truth: {
                layer: {k: _json_value(v) for k, v in m.as_dict().items()}
                for layer, m in layers.items()
            }
            for truth, layers in r.results.items()
        },
    }
    path = Path("reports") / f"oracle-{r.source}-s{r.scene}-c{r.camera}-{r.cell_m}m.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path


def _print_camera(r, path):
    print(f"{r.dataset}  scene {r.scene}  camera {r.camera}  source {r.source}  cell {r.cell_m} m")
    print(
        f"samples {r.samples}  outside grid {r.dropped}  coverage cells {r.coverage_cells}  "
        f"visibility cells {r.visibility_cells}  occlusion fraction {r.occlusion_fraction:.4f}"
    )
    for truth in TRUTHS:
        print(f"\n{truth}: {TRUTH_TITLES[truth]}")
        print(f"{'layer':<8}" + "".join(f"{m:>12}" for m in METRICS))
        for layer in LAYERS:
            m = r.results[truth][layer]
            print(f"{layer:<8}" + "".join(_metric(m, name) for name in METRICS))
    print(f"\nreport {path}")


# --camera all: one row per camera. The primary block in full, then the dwell
# TVD of every block in TRUTHS order.
SHORT = {"visible@coverage": "vcov", "full@coverage": "fcov", "visible@visibility": "vvis"}
ALL_METRICS = [
    ("visible@coverage", "dwell", "mass_error"),
    ("visible@coverage", "dwell", "tvd"),
    ("visible@coverage", "dwell", "wape_zone"),
    ("visible@coverage", "dwell", "emd_m"),
    ("visible@coverage", "passage", "mass_error"),
    ("visible@coverage", "passage", "tvd"),
    ("full@coverage", "dwell", "tvd"),
    ("visible@visibility", "dwell", "tvd"),
]
ALL_HEADER = ["coverage", "visibility", "occlusion"] + [
    f"{SHORT[t]}.{layer[0]}.{m[:4].rstrip('_')}" for t, layer, m in ALL_METRICS
]


# Per-camera failures of --camera all: domain errors (ValueError) and missing
# data (a key absent from a file, a file absent from disk). Anything else is a bug.
CAMERA_ERRORS = (ValueError, KeyError, OSError)


def _all_row(r):
    values = [r.coverage_cells, r.visibility_cells]
    cells = "".join(f"{v:>12}" for v in values) + _cell(r.occlusion_fraction, 12)
    for t, layer, m in ALL_METRICS:
        cells += _metric(r.results[t][layer], m)
    return f"{r.camera:<8}{cells}"


def run_eval(args):
    reader = NvidiaMtmcReader(data_dir())
    if args.camera != "all":
        r = evaluate_oracle(reader, args.source, args.scene, args.camera, args.cell)
        _print_camera(r, _write_report(r))
        return 0

    print(f"{reader.name}  scene {args.scene}  source {args.source}  cell {args.cell} m")
    print(
        "columns: vcov = visible@coverage (E1 primary), fcov = full@coverage (E2a target), "
        "vvis = visible@visibility (diagnostic); d/p = dwell/passage"
    )
    print(f"{'camera':<8}" + "".join(f"{name:>12}" for name in ALL_HEADER))
    code = 0
    for camera in reader.cameras(args.scene):
        try:
            r = evaluate_oracle(reader, args.source, args.scene, camera, args.cell)
        except CAMERA_ERRORS as e:
            print(f"{camera:<8}error: {type(e).__name__}: {e}")
            code = 1
            continue
        _write_report(r)
        print(_all_row(r))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="space-usage")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="score a heatmap source against ground truth")
    ev.add_argument("--source", choices=SOURCES, required=True)
    ev.add_argument("--scene", required=True, help="scene id, e.g. 071")
    ev.add_argument("--camera", required=True, help="camera id, e.g. 645, or all")
    ev.add_argument("--cell", type=float, choices=[0.25, 0.5, 1.0], default=0.5)
    args = parser.parse_args(argv)
    try:
        return run_eval(args)
    except CAMERA_ERRORS as e:
        print(f"space-usage: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
