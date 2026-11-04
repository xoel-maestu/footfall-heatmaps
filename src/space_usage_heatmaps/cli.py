"""space-usage: command-line entry point (driving adapter)."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import yaml

from space_usage_heatmaps.evaluation.application.evaluate_config import evaluate_config
from space_usage_heatmaps.evaluation.application.evaluate_oracle import (
    SOURCES,
    TRUTHS,
    evaluate_oracle,
)
from space_usage_heatmaps.evaluation.domain.run_config import RunConfig
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.evaluation.infrastructure.track_cache import CachedTrackSource
from space_usage_heatmaps.occupancy.domain.grid import GridSpec

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
    if args.config:
        return run_eval_config(args)
    if not (args.scene and args.camera):
        print("space-usage: error: --source needs --scene and --camera", file=sys.stderr)
        return 2
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


BASELINE_MISSING = (
    "the iteration 0 baseline needs the optional dependency group: uv sync --group baseline"
)


def load_config(path):
    return RunConfig.from_mapping(yaml.safe_load(Path(path).read_text()))


def cache_key(c):
    """Cached tracks depend on everything but the foot point, applied when scoring."""
    return f"{c.detector}-{c.tracker}-{c.input_size}-{c.engine}-stride{c.stride}"


def track_source(config, reader):
    def make():
        try:
            from space_usage_heatmaps.evaluation.baseline.ultralytics_tracks import (
                UltralyticsTrackSource,
            )
        except ImportError as e:
            raise OSError(BASELINE_MISSING) from e
        return UltralyticsTrackSource(config, reader.video, data_dir() / "models")

    return CachedTrackSource(make, data_dir() / "cache" / "tracks" / cache_key(config))


def config_name(path):
    """Report name of a config: its path relative to the working directory (the
    repo), suffix dropped, separators replaced, so configs/a/x.yaml and
    configs/b/x.yaml do not collide."""
    p = Path(path).absolute()
    try:
        p = p.relative_to(Path.cwd())
    except ValueError:
        p = p.relative_to(p.anchor)
    return "_".join(p.with_suffix("").parts)


INTERVAL = (
    "recentred percentile: percentiles of the resamples shifted so their mean is the "
    "whole-run value; absolute intervals clipped to each metric's domain, deltas not"
)


def _ci(e, width=22):
    if e.value is None:
        return f"{'n/a':>{width}}"
    if e.lo is None:
        return f"{e.value:>{width}.3f}"
    return f"{f'{e.value:.3f} [{e.lo:.3f}, {e.hi:.3f}]':>{width}}"


def _rows(label, per_scene, pooled, scenes, width):
    total = sum(run.blocks for run in scenes)
    for run, table in [*zip(scenes, per_scene, strict=True), (None, pooled)]:
        where = "pooled" if run is None else f"{run.scene}/{run.camera}"
        blocks = total if run is None else run.blocks
        for layer in LAYERS:
            cells = "".join(_ci(table[layer][m]) for m in METRICS)
            print(f"{label:<{width}}{where:<10}{layer:<9}{blocks:>7}{cells}")


def _estimates(table):
    return {layer: {m: vars(e) for m, e in metrics.items()} for layer, metrics in table.items()}


def run_eval_config(args):
    if not args.split:
        print("space-usage: error: --config needs --split", file=sys.stderr)
        return 2
    reader = NvidiaMtmcReader(data_dir())
    names = [config_name(args.config)] + ([config_name(args.compare)] if args.compare else [])
    configs = [load_config(args.config)] + ([load_config(args.compare)] if args.compare else [])
    runs = [(c, track_source(c, reader)) for c in configs]
    r = evaluate_config(reader, SPLITS[args.split], runs, args.cell, n_resamples=args.resamples)

    print(
        f"{r.dataset}  split {args.split}  cell {r.cell_m} m  "
        f"paired block bootstrap: {r.block_s:g} s blocks, {r.n_resamples} resamples, "
        "95 % recentred percentile CI (clipped to the metric's domain; deltas not clipped)"
    )
    print(
        "pooled = mean over scenes, blocks resampled within each scene"
        + (f"; delta = {' - '.join(names)}, paired" if args.compare else "")
    )
    for run in r.scenes:
        print(
            f"scene {run.scene} camera {run.camera}: {run.samples} tracked boxes, "
            f"{run.dropped} outside grid, {run.outside_truth_frames} outside truth frames, "
            f"{run.blocks} blocks, occlusion fraction {run.occlusion_fraction:.4f}"
        )
    width = max(len(n) for n in [*names, "config"]) + 2
    n_blocks = {
        "scenes": {f"{run.scene}/{run.camera}": run.blocks for run in r.scenes},
        "pooled": sum(run.blocks for run in r.scenes),
    }
    for truth in TRUTHS:
        res = r.results[truth]
        print(f"\n{truth}: {TRUTH_TITLES[truth]}")
        print(
            f"{'config':<{width}}{'scene':<10}{'layer':<9}{'blocks':>7}"
            + "".join(f"{m:>22}" for m in METRICS)
        )
        for i, name in enumerate(names):
            _rows(name, [s[i] for s in res.scenes], res.pooled[i], r.scenes, width)
        if res.delta_pooled is not None:
            _rows("delta", res.delta_scenes, res.delta_pooled, r.scenes, width)

    report = {
        "dataset": r.dataset,
        "split": args.split,
        "cell_m": r.cell_m,
        "block_s": r.block_s,
        "n_resamples": r.n_resamples,
        "level": 0.95,
        "interval": INTERVAL,
        "pooled": "unweighted mean over scenes, blocks resampled within each scene",
        "configs": {n: vars(c) for n, c in zip(names, configs, strict=True)},
        "scenes": [vars(run) for run in r.scenes],
        "results": {
            truth: {
                "n_blocks": n_blocks,
                "scenes": {
                    f"{run.scene}/{run.camera}": {
                        n: _estimates(t) for n, t in zip(names, tables, strict=True)
                    }
                    for run, tables in zip(r.scenes, res.scenes, strict=True)
                },
                "pooled": {n: _estimates(t) for n, t in zip(names, res.pooled, strict=True)},
            }
            | (
                {}
                if res.delta_pooled is None
                else {
                    "delta": {
                        "of": names,
                        "scenes": {
                            f"{run.scene}/{run.camera}": _estimates(t)
                            for run, t in zip(r.scenes, res.delta_scenes, strict=True)
                        },
                        "pooled": _estimates(res.delta_pooled),
                    }
                }
            )
            for truth, res in r.results.items()
        },
    }
    name = "-vs-".join(names)
    path = Path("reports") / f"eval-{name}-{args.split}-{r.cell_m}m.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nreport {path}")
    return 0


BENCH_SCENE, BENCH_CAMERA, BENCH_CELL = "071", "645", 0.25


def run_bench(args):
    """End to end on scene 071's 1080p video, looped for --minutes."""
    config = load_config(args.config)
    reader = NvidiaMtmcReader(data_dir())
    cam = reader.camera(BENCH_SCENE, BENCH_CAMERA)
    spec = GridSpec.covering(*reader.scene_extent(BENCH_SCENE), cell_m=BENCH_CELL)
    try:
        from space_usage_heatmaps.evaluation.baseline.bench import bench
    except ImportError as e:
        raise OSError(BASELINE_MISSING) from e
    result = bench(
        config,
        reader.video(BENCH_SCENE, BENCH_CAMERA),
        cam.world_to_image,
        spec,
        cam.fps,
        args.minutes,
        data_dir() / "models",
    )
    print(json.dumps(result))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="space-usage")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="score a heatmap source against ground truth")
    how = ev.add_mutually_exclusive_group(required=True)
    how.add_argument("--source", choices=SOURCES, help="oracle: ground truth as the prediction")
    how.add_argument("--config", help="run configuration, e.g. configs/iter0.yaml")
    ev.add_argument("--scene", help="with --source: scene id, e.g. 071")
    ev.add_argument("--camera", help="with --source: camera id, e.g. 645, or all")
    ev.add_argument("--split", choices=sorted(SPLITS), help="with --config: scenes to score")
    ev.add_argument("--compare", help="with --config: second configuration, paired difference")
    ev.add_argument("--resamples", type=int, default=1000, help=argparse.SUPPRESS)
    ev.add_argument("--cell", type=float, choices=[0.25, 0.5, 1.0], default=0.5)
    be = sub.add_parser("bench", help="sustained end-to-end fps and peak VRAM on one video")
    be.add_argument("--config", required=True, help="run configuration, e.g. configs/iter0.yaml")
    be.add_argument("--minutes", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        return run_bench(args) if args.command == "bench" else run_eval(args)
    except CAMERA_ERRORS as e:
        print(f"space-usage: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
