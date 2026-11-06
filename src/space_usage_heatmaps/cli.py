"""space-usage: command-line entry point (driving adapter)."""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
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
from space_usage_heatmaps.floorplan.domain.foot_point import FOOT_POINTS
from space_usage_heatmaps.occupancy.application.purge_slots import RETENTION_DAYS, purge_slots
from space_usage_heatmaps.occupancy.application.record_camera import record_camera
from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
from space_usage_heatmaps.perception.application.reconnect import reconnecting_frames
from space_usage_heatmaps.perception.domain.privacy_mask import PrivacyMask
from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource

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
        "foot_point": r.foot_point,
        "head_samples": r.head_samples,
        "position_error_m": r.position_error.as_dict(),
        "results": {
            truth: {
                layer: {k: _json_value(v) for k, v in m.as_dict().items()}
                for layer, m in layers.items()
            }
            for truth, layers in r.results.items()
        },
    }
    source = r.source if r.foot_point is None else f"{r.source}-{r.foot_point}"
    path = Path("reports") / f"oracle-{source}-s{r.scene}-c{r.camera}-{r.cell_m}m.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path


def _source(r):
    return r.source if r.foot_point is None else f"{r.source}  foot point {r.foot_point}"


def _position(e):
    return f"{_cell(e.median, 0)} m median, {_cell(e.p90, 0)} m p90 over {e.matched} samples"


def _print_camera(r, path):
    print(
        f"{r.dataset}  scene {r.scene}  camera {r.camera}  source {_source(r)}  cell {r.cell_m} m"
    )
    print(
        f"samples {r.samples}  outside grid {r.dropped}  coverage cells {r.coverage_cells}  "
        f"visibility cells {r.visibility_cells}  occlusion fraction {r.occlusion_fraction:.4f}"
    )
    print(
        f"position error against the 2025 JSON truth: {_position(r.position_error)} "
        f"({r.position_error.unmatched} unmatched); from the box top: {r.head_samples} samples"
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
ALL_HEADER = (
    ["coverage", "visibility", "occlusion"]
    + [f"{SHORT[t]}.{layer[0]}.{m[:4].rstrip('_')}" for t, layer, m in ALL_METRICS]
    + ["pos.med", "pos.p90"]
)


# Per-camera failures of --camera all: domain errors (ValueError) and missing
# data (a key absent from a file, a file absent from disk). Anything else is a bug.
CAMERA_ERRORS = (ValueError, KeyError, OSError)


def _all_row(r):
    values = [r.coverage_cells, r.visibility_cells]
    cells = "".join(f"{v:>12}" for v in values) + _cell(r.occlusion_fraction, 12)
    for t, layer, m in ALL_METRICS:
        cells += _metric(r.results[t][layer], m)
    cells += _cell(r.position_error.median) + _cell(r.position_error.p90)
    return f"{r.camera:<8}{cells}"


def run_eval(args):
    if args.config:
        return run_eval_config(args)
    if not (args.scene and args.camera):
        print("space-usage: error: --source needs --scene and --camera", file=sys.stderr)
        return 2
    if args.foot_point and args.source != "gt-boxes":
        print("space-usage: error: --foot-point applies to --source gt-boxes", file=sys.stderr)
        return 2
    foot = args.foot_point or "bottom"
    reader = NvidiaMtmcReader(data_dir())
    if args.camera != "all":
        r = evaluate_oracle(reader, args.source, args.scene, args.camera, args.cell, foot)
        _print_camera(r, _write_report(r))
        return 0

    shown = args.source if args.source == "gt-world" else f"{args.source}  foot point {foot}"
    print(f"{reader.name}  scene {args.scene}  source {shown}  cell {args.cell} m")
    print(
        "columns: vcov = visible@coverage (E1 primary), fcov = full@coverage (E2a target), "
        "vvis = visible@visibility (diagnostic); d/p = dwell/passage; "
        "pos = per-sample ground position error in m"
    )
    print(f"{'camera':<8}" + "".join(f"{name:>12}" for name in ALL_HEADER))
    code = 0
    for camera in reader.cameras(args.scene):
        try:
            r = evaluate_oracle(reader, args.source, args.scene, camera, args.cell, foot)
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
    """The pipeline recipe of a config; its run section is space-usage run's."""
    d = yaml.safe_load(Path(path).read_text())
    return RunConfig.from_mapping({k: v for k, v in d.items() if k != "run"})


@dataclass(frozen=True)
class RunSettings:
    """A config's run section: the one camera space-usage run records."""

    camera_id: str
    calibration: str  # nvidia-mtmc/<scene>/<camera>, under the data dir
    extent_m: tuple  # floor grid: xmin, ymin, xmax, ymax in metres
    cell_m: float = 0.5
    retention_days: float = RETENTION_DAYS
    masks: dict = field(default_factory=dict)  # image and floor polygons (R6)
    start_utc: object = None  # a video file's start; --start overrides


def load_run_settings(path):
    d = (yaml.safe_load(Path(path).read_text()) or {}).get("run")
    if not isinstance(d, dict):
        raise ValueError(f"config {path}: no run section (camera_id, calibration, extent_m)")
    names = {f.name for f in fields(RunSettings)}
    if unknown := sorted(set(d) - names):
        raise ValueError(f"config {path}: unknown run keys {unknown}")
    s = RunSettings(**d)
    if not isinstance(s.extent_m, list | tuple) or len(s.extent_m) != 4:
        raise ValueError(f"config {path}: run.extent_m must be [xmin, ymin, xmax, ymax]")
    if unknown := sorted(set(s.masks) - {"image", "floor"}):
        raise ValueError(f"config {path}: run.masks takes image and floor, not {unknown}")
    if not s.retention_days > 0:
        raise ValueError(f"config {path}: run.retention_days must be positive")
    return replace(s, camera_id=str(s.camera_id))


def utc_epoch(value):
    """Epoch seconds of an ISO 8601 time with a zone (or a YAML timestamp, which is UTC)."""
    if isinstance(value, datetime):
        t = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        t = datetime.fromisoformat(str(value))
        if t.tzinfo is None:
            raise ValueError(f"start {value!r} needs a time zone, e.g. 2026-11-06T09:00:00Z")
    return t.timestamp()


def calibration_of(spec):
    dataset, scene, camera = (spec.split("/") + ["", ""])[:3]
    if dataset != "nvidia-mtmc" or not scene or not camera:
        raise ValueError(f"calibration {spec!r}: expected nvidia-mtmc/<scene>/<camera>")
    return NvidiaMtmcReader(data_dir()).camera(scene, camera)


STOP_WAIT_S = 10  # serve: how long the recorder gets to write its open slot on exit


def run_record(args, live=None, stop=None):
    """space-usage run: one camera into 15-min slots in SQLite, with the live layer
    in memory. A video file is read once from --start (or run.start_utc, or now);
    a stream (rtsp://) is timed by the wall clock and reconnects when it drops.
    live: a LiveSource the viewer reads the live layer from (space-usage serve);
    stop: an Event that ends the frames, so the open slot is still written."""
    from space_usage_heatmaps.perception.infrastructure import container
    from space_usage_heatmaps.perception.infrastructure.opencv_stream_source import (
        OpenCvStreamSource,
    )

    config, settings = load_config(args.config), load_run_settings(args.config)
    cam = calibration_of(settings.calibration)
    spec = GridSpec.covering(*settings.extent_m, cell_m=settings.cell_m)
    mask = PrivacyMask(
        settings.masks.get("image") or (),
        settings.masks.get("floor") or (),
        cam.world_to_image,
    )
    store = SqliteSlotStore(args.db)
    recorder = RecordSlots(store, settings.camera_id, spec, settings.retention_days, time.time)
    if live is not None:
        recorder = live.attach(recorder)
    purge_slots(store, time.time(), settings.retention_days)
    detector = container.build_detector(
        config.detector,
        config.input_size,
        data_dir() / "models",
        config.detector_params,
        config.engine,
    )
    if args.source.startswith("rtsp://"):
        first = [OpenCvStreamSource(args.source, config.stride)]
        fps = first[0].fps

        def open_stream():
            return first.pop() if first else OpenCvStreamSource(args.source, config.stride)

        def time_of(frame):
            return time.time()

        frames = reconnecting_frames(open_stream, time.sleep, recorder.mark_incomplete)
    else:
        source = OpenCvFileSource(args.source, config.stride)
        fps, frames = source.fps, source.frames()
        start = args.start or settings.start_utc
        t0 = time.time() if start is None else utc_epoch(start)

        def time_of(frame):
            return t0 + frame.timestamp

    tracker = container.build_tracker(
        config.tracker, fps / config.stride, cam.world_to_image, config.tracker_params
    )
    if stop is not None:
        frames = _until(stop, frames)
    try:
        n = record_camera(
            frames, detector, mask, tracker, cam, config, recorder, time_of, config.stride / fps
        )
    finally:
        store.close()
    print(
        f"camera {settings.camera_id}: {n} frames, {recorder.written} slots "
        f"({spec.ny}x{spec.nx} cells of {settings.cell_m} m) in {args.db}"
    )
    if recorder.past_retention:
        print(
            f"camera {settings.camera_id}: {recorder.past_retention} slots past retention "
            f"({settings.retention_days} days) were not written; check --start",
            file=sys.stderr,
        )
        return 1
    return 0


def _until(stop, frames):
    for frame in frames:
        if stop.is_set():
            return
        yield frame


def start_serve(args):
    """The viewer's server for the config's camera, and with --source the recorder
    in a thread of this process: the live layer stays in its memory, and the
    server reads it there. (server, recorder thread or None, stop event)."""
    from space_usage_heatmaps.web.server import LiveSource, make_server

    settings = load_run_settings(args.config)
    spec = GridSpec.covering(*settings.extent_m, cell_m=settings.cell_m)
    live = LiveSource() if args.source else None
    server = make_server(args.host, args.port, settings.camera_id, spec, args.db, live)
    stop, recording = threading.Event(), None
    if args.source:

        def record():
            try:
                run_record(args, live, stop)
            except CAMERA_ERRORS as e:
                print(f"space-usage: error: {e}", file=sys.stderr)

        recording = threading.Thread(target=record, name="recorder", daemon=True)
        recording.start()
    return server, recording, stop


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def run_serve(args):
    """space-usage serve: the page, /api/live and /api/slots until Ctrl+C or SIGTERM."""
    server, recording, stop = start_serve(args)
    host, port = server.server_address[:2]
    live = f"live from {args.source}" if args.source else "slots only, no live source"
    print(f"space-usage serve: http://{host}:{port}/ ({live})", flush=True)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stop.set()
        if recording is not None:
            recording.join(timeout=STOP_WAIT_S)
    return 0


def _params_suffix(params):
    data = json.dumps(params, sort_keys=True).encode()
    return f"-params{hashlib.sha256(data).hexdigest()[:12]}"


def cache_key(c):
    """Directory of a config's cached tracks; each file also stores the recipe
    metadata and is recomputed when it no longer matches (track_cache). Tracker
    and detector overrides get their own directory, so variants do not
    overwrite each other."""
    key = f"{c.detector}-{c.tracker}-{c.input_size}-{c.engine}-stride{c.stride}"
    if c.detector_params:
        return key + _params_suffix({"tracker": c.tracker_params, "detector": c.detector_params})
    return key + _params_suffix(c.tracker_params) if c.tracker_params else key


def detection_cache_key(c):
    """Directory of a detector recipe's cached detections (detection_cache):
    every tracker, stride and higher score threshold of that recipe replays
    them, so they hold every frame at the cache's score floor."""
    from space_usage_heatmaps.perception.infrastructure.container import cache_params

    key = f"{c.detector}-{c.input_size}-{c.engine}-stride1"
    params = cache_params(c.detector, c.detector_params)
    return key + _params_suffix(params) if params else key


def pipeline_of(config, reader, loop=False):
    """(scene, camera) -> the perception pipeline of a config on that camera's video."""
    from space_usage_heatmaps.perception.infrastructure.container import build_pipeline

    def build(scene, camera):
        return build_pipeline(
            reader.video(scene, camera),
            config.detector,
            config.tracker,
            config.input_size,
            config.stride,
            data_dir() / "models",
            loop,
            reader.camera(scene, camera).world_to_image,
            config.tracker_params,
            config.detector_params,
            config.engine,
        )

    return build


def pipeline_track_source(config, reader):
    """Tracks of a pipeline config: cached detections of its detector recipe on
    every frame, its stride and score threshold applied to them, then a fresh
    tracker on each video and the skipped frames filled."""
    import numpy as np

    from space_usage_heatmaps.evaluation.infrastructure.detection_cache import CachedDetections
    from space_usage_heatmaps.evaluation.infrastructure.pipeline_track_source import (
        PipelineDetections,
        PipelineTrackSource,
        SelectedDetections,
    )
    from space_usage_heatmaps.perception.infrastructure import container

    cached = container.cache_params(config.detector, config.detector_params)

    def detection_of(scene, camera):
        return container.build_detection(
            reader.video(scene, camera),
            config.detector,
            config.input_size,
            1,
            data_dir() / "models",
            params=cached,
            engine=config.engine,
        )

    def tracker_of(scene, camera):
        return container.build_tracker(
            config.tracker,
            container.video_fps(reader.video(scene, camera)) / config.stride,
            reader.camera(scene, camera).world_to_image,
            config.tracker_params,
        )

    def video_identity(scene, camera):
        return {"video_bytes": reader.video(scene, camera).stat().st_size}

    def track_identity(scene, camera):
        """The video, and for a tracker on the floor the camera's homography."""
        if config.tracker != "ucmc":
            return video_identity(scene, camera)
        h = np.ascontiguousarray(reader.camera(scene, camera).world_to_image, dtype=float)
        return video_identity(scene, camera) | {
            "homography_sha256": hashlib.sha256(h.tobytes()).hexdigest()
        }

    detections = CachedDetections(
        PipelineDetections(detection_of),
        data_dir() / "cache" / "detections" / detection_cache_key(config),
        container.detection_metadata(config.detector, config.input_size, 1, cached, config.engine),
        identity=video_identity,
    )
    meta = container.pipeline_metadata(
        config.detector,
        config.tracker,
        config.input_size,
        config.stride,
        config.tracker_params,
        config.detector_params,
        config.engine,
    )
    directory = data_dir() / "cache" / "tracks" / cache_key(config)
    selected = SelectedDetections(
        detections, config.stride, container.min_score(config.detector, config.detector_params)
    )
    return CachedTrackSource(
        lambda: PipelineTrackSource(selected, tracker_of, config.stride),
        directory,
        meta,
        identity=track_identity,
    )


def track_source(config, reader):
    directory = data_dir() / "cache" / "tracks" / cache_key(config)
    if config.engine != "ultralytics":
        return pipeline_track_source(config, reader)

    def make():
        try:
            from space_usage_heatmaps.evaluation.baseline.ultralytics_tracks import (
                UltralyticsTrackSource,
            )
        except ImportError as e:
            raise OSError(BASELINE_MISSING) from e
        return UltralyticsTrackSource(config, reader.video, data_dir() / "models")

    from space_usage_heatmaps.evaluation.baseline.recipe import recipe_metadata

    return CachedTrackSource(make, directory, recipe_metadata(config))


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


# Spec 6 guard: HOTA and IDF1 may not drop more than this against the config compared to.
GUARD_POINTS = 1.0
GUARD_BROKEN = 3  # eval's exit code when the guard breaks (the report is still written)
TRACKING = ["hota", "deta", "assa", "idf1"]


def score_tracking(reader, scenes, names, runs):
    """HOTA, DetA, AssA and IDF1 per config and scene in image space against the
    camera's truth boxes, pooled as the mean over scenes; with two configs, the
    difference first - second."""
    from space_usage_heatmaps.evaluation.infrastructure.tracking_metrics import (
        pooled,
        tracking_metrics,
    )

    keys = [f"{scene}/{camera}" for scene, camera in scenes]
    per = {n: {} for n in names}
    for (scene, camera), key in zip(scenes, keys, strict=True):
        samples = reader.samples(scene, camera)
        for n, (_, source) in zip(names, runs, strict=True):
            per[n][key] = tracking_metrics(samples, source.tracks(scene, camera))
    result = {"scenes": per, "pooled": {n: pooled(list(per[n].values())) for n in names}}
    if len(names) == 2:
        a, b = names

        def diff(x, y):
            return {m: x[m] - y[m] for m in TRACKING}

        result["delta"] = {
            "of": names,
            "scenes": {k: diff(per[a][k], per[b][k]) for k in keys},
            "pooled": diff(result["pooled"][a], result["pooled"][b]),
        }
    return result


def guard_holds(t):
    """Whether HOTA and IDF1 dropped at most GUARD_POINTS; None without a comparison."""
    if "delta" not in t:
        return None
    d = t["delta"]["pooled"]
    return 100 * min(d["hota"], d["idf1"]) >= -GUARD_POINTS


def _print_tracking(t, names, width):
    print(
        "\ntracking guards: HOTA (DetA, AssA) and IDF1 in image space against the "
        "camera's truth boxes, TrackEval definitions; pooled = mean over scenes"
    )
    print(f"{'config':<{width}}{'scene':<10}{'':<9}" + "".join(f"{m:>8}" for m in TRACKING))
    rows = [(n, t["scenes"][n], t["pooled"][n]) for n in names]
    if "delta" in t:
        rows.append(("delta", t["delta"]["scenes"], t["delta"]["pooled"]))
    for label, scenes, pool in rows:
        for where, m in [*scenes.items(), ("pooled", pool)]:
            cells = "".join(f"{m[k]:>8.3f}" for k in TRACKING)
            print(f"{label:<{width}}{where:<10}{'tracking':<9}{cells}")
    if "delta" in t:
        d = {k: 100 * v for k, v in t["delta"]["pooled"].items()}
        holds = guard_holds(t)
        print(
            f"guard: HOTA {d['hota']:+.1f} points, IDF1 {d['idf1']:+.1f} points: "
            + ("holds" if holds else f"BROKEN (more than {GUARD_POINTS:g} point lower)")
        )


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
    tracking = score_tracking(reader, SPLITS[args.split], names, runs)
    tracking["guard_holds"] = guard_holds(tracking)

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
    _print_tracking(tracking, names, width)

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
        "tracking": tracking,
        "tracks_provenance": {
            n: getattr(src, "provenance", None) for n, (_, src) in zip(names, runs, strict=True)
        },
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
    if tracking["guard_holds"] is False:
        print("space-usage: tracking guard broken", file=sys.stderr)
        return GUARD_BROKEN
    return 0


BENCH_SCENE, BENCH_CAMERA, BENCH_CELL = "071", "645", 0.25


RUNTIME_MISSING = "the perception pipeline needs the runtime group: uv sync --group runtime"


def engine_label(engine, providers, ort):
    """What ran: the config's engine, the providers in order, the precision of the first."""
    precision = "fp16" if providers[0] == "TensorrtExecutionProvider" else "fp32"
    return f"{engine} (ONNX Runtime {ort}, {' then '.join(providers)}, {precision})"


def bench_pipeline(config, reader, cam, spec, minutes):
    from importlib import metadata

    from space_usage_heatmaps.perception.application.track_video import fill_stride, track_video

    try:
        from space_usage_heatmaps.evaluation.infrastructure import bench as sustained

        ort = metadata.version("onnxruntime-gpu")
    except (ImportError, metadata.PackageNotFoundError) as e:
        raise OSError(RUNTIME_MISSING) from e
    source, detector, tracker = pipeline_of(config, reader, loop=True)(BENCH_SCENE, BENCH_CAMERA)
    tracked = (
        (t.frame_index, t.track_ids, t.boxes)
        for t in fill_stride(track_video(source, detector, tracker), config.stride)
    )
    result = sustained.bench(tracked, config, cam, spec, minutes)
    return {"engine": engine_label(config.engine, detector.providers, ort)} | result


def run_bench(args):
    """End to end on scene 071's 1080p video, looped for --minutes."""
    config = load_config(args.config)
    reader = NvidiaMtmcReader(data_dir())
    cam = reader.camera(BENCH_SCENE, BENCH_CAMERA)
    spec = GridSpec.covering(*reader.scene_extent(BENCH_SCENE), cell_m=BENCH_CELL)
    if config.engine != "ultralytics":
        result = bench_pipeline(config, reader, cam, spec, args.minutes)
        print(json.dumps(result))
        return 0
    try:
        from space_usage_heatmaps.evaluation.baseline.bench import bench
    except ImportError as e:
        raise OSError(BASELINE_MISSING) from e
    result = bench(
        config,
        reader.video(BENCH_SCENE, BENCH_CAMERA),
        cam,
        spec,
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
    ev.add_argument(
        "--foot-point", choices=FOOT_POINTS, help="with --source gt-boxes: default bottom"
    )
    ev.add_argument("--split", choices=sorted(SPLITS), help="with --config: scenes to score")
    ev.add_argument("--compare", help="with --config: second configuration, paired difference")
    ev.add_argument("--resamples", type=int, default=1000, help=argparse.SUPPRESS)
    ev.add_argument("--cell", type=float, choices=[0.25, 0.5, 1.0], default=0.5)
    be = sub.add_parser("bench", help="sustained end-to-end fps and peak VRAM on one video")
    be.add_argument("--config", required=True, help="run configuration, e.g. configs/iter0.yaml")
    be.add_argument("--minutes", type=float, default=30.0)
    ru = sub.add_parser("run", help="record a camera into persistent 15-min slots")
    ru.add_argument("--config", required=True, help="configuration with a run section")
    ru.add_argument("--source", required=True, help="video file or rtsp:// stream")
    ru.add_argument("--db", required=True, help="SQLite file of the slots")
    ru.add_argument("--start", help="video file: UTC start, e.g. 2026-11-06T09:10:00Z")
    se = sub.add_parser("serve", help="viewer page with the live layer and the persisted slots")
    se.add_argument("--config", required=True, help="configuration with a run section")
    se.add_argument("--db", required=True, help="SQLite file of the slots")
    se.add_argument("--source", help="video file or rtsp:// stream to record in this process")
    se.add_argument("--start", help="video file: UTC start, e.g. 2026-11-06T09:10:00Z")
    se.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    se.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    commands = {"bench": run_bench, "eval": run_eval, "run": run_record, "serve": run_serve}
    try:
        return commands[args.command](args)
    except CAMERA_ERRORS as e:
        print(f"space-usage: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
