"""space-usage: command-line entry point (driving adapter)."""

import argparse
import hashlib
import ipaddress
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
from types import SimpleNamespace

import yaml

from space_usage_heatmaps.evaluation.application.evaluate_config import evaluate_config
from space_usage_heatmaps.evaluation.application.evaluate_oracle import (
    SOURCES,
    TRUTHS,
    evaluate_oracle,
    stitched_tracks,
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


SUITES = ("e1-final",)


def run_eval(args):
    if args.metric == "mc-idf1":
        if not args.cameras:
            print("space-usage: error: --metric mc-idf1 goes with --cameras", file=sys.stderr)
            return 2
        if args.source or args.suite or args.compare:
            print("space-usage: error: --metric mc-idf1 takes --config only", file=sys.stderr)
            return 2
        if args.dataset not in (None, "wildtrack"):
            print("space-usage: error: --metric mc-idf1 takes --dataset wildtrack", file=sys.stderr)
            return 2
        if args.dataset is None and bool(args.scene) == bool(args.split):
            print("space-usage: error: --cameras needs --scene or --split", file=sys.stderr)
            return 2
        return run_eval_identity(args)
    if args.cameras:
        if args.source or args.suite or args.dataset:
            print("space-usage: error: --cameras goes with --config only", file=sys.stderr)
            return 2
        if bool(args.scene) == bool(args.split):
            print("space-usage: error: --cameras needs --scene or --split", file=sys.stderr)
            return 2
        return run_eval_fusion(args)
    if not (args.source or args.config or args.suite):
        print(
            "space-usage: error: one of --source, --config, --suite or --cameras", file=sys.stderr
        )
        return 2
    if args.suite:
        from space_usage_heatmaps.e1_final import run_e1_final

        return run_e1_final(args)
    if args.dataset:
        if not args.config:
            print("space-usage: error: --dataset needs --config", file=sys.stderr)
            return 2
        return run_eval_dataset(args)
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
    """The pipeline recipe of a config; its run section is space-usage run's,
    its fusion section eval --cameras' (load_fusion), its association section
    eval --metric mc-idf1's (load_association)."""
    d = yaml.safe_load(Path(path).read_text())
    sections = ("run", "fusion", "association")
    return RunConfig.from_mapping({k: v for k, v in d.items() if k not in sections})


def load_fusion(path):
    """A config's fusion section: weighting, power (.inf allowed), zone_m."""
    from space_usage_heatmaps.occupancy.domain.fusion import Fusion

    d = (yaml.safe_load(Path(path).read_text()) or {}).get("fusion")
    if not isinstance(d, dict):
        raise ValueError(f"config {path}: no fusion section (weighting, power, zone_m)")
    if unknown := sorted(set(d) - {"weighting", "power", "zone_m"}):
        raise ValueError(f"config {path}: unknown fusion keys {unknown}")
    return Fusion(**d)


def load_association(path):
    """A config's association section (occupancy.domain.association.Association)."""
    from dataclasses import fields as dataclass_fields

    from space_usage_heatmaps.occupancy.domain.association import Association

    d = (yaml.safe_load(Path(path).read_text()) or {}).get("association")
    if not isinstance(d, dict):
        raise ValueError(f"config {path}: no association section (method, gate_m, ...)")
    if unknown := sorted(set(d) - {f.name for f in dataclass_fields(Association)}):
        raise ValueError(f"config {path}: unknown association keys {unknown}")
    return Association(**d)


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
    if config.stitch is not None:
        raise ValueError(
            f"config {args.config}: stitch joins tracks across later frames, an evaluation "
            "step; space-usage run records the tracker's own ids"
        )
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


def _is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def start_serve(args):
    """The viewer's server for the config's camera, and with --source the recorder
    in a thread of this process: the live layer stays in its memory, and the
    server reads it there. (server, recorder thread or None, stop event)."""
    from space_usage_heatmaps.web.server import LiveSource, make_server

    settings = load_run_settings(args.config)
    spec = GridSpec.covering(*settings.extent_m, cell_m=settings.cell_m)
    live = LiveSource() if args.source else None
    fused = json.loads(Path(args.fused).read_text()) if args.fused else None
    server = make_server(args.host, args.port, settings.camera_id, spec, args.db, live, fused=fused)
    if not _is_loopback(args.host):
        print(
            f"space-usage serve: warning: --host {args.host} opens the page and the live layer "
            "to other machines, with no authentication",
            file=sys.stderr,
        )
    stop, recording = threading.Event(), None
    if args.source:

        def record():
            try:
                run_record(args, live, stop)
                if not args.source.startswith("rtsp://") and not stop.is_set():
                    live.end(time.time())
            except CAMERA_ERRORS as e:
                live.fail(e)
                print(f"space-usage: error: {e}", file=sys.stderr)
            except Exception as e:  # any other failure: /api/live answers 503, not a fading layer
                live.fail(e)
                raise

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


def detection_cache_key(c, stride=1):
    """Directory of a detector recipe's cached detections (detection_cache):
    every tracker, stride and higher score threshold of that recipe replays
    them, so they hold every frame at the cache's score floor. With a stride,
    those of every stride-th frame only, for a camera without every-frame ones."""
    from space_usage_heatmaps.perception.infrastructure.container import cache_params

    key = f"{c.detector}-{c.input_size}-{c.engine}-stride{stride}"
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

    from space_usage_heatmaps.evaluation.infrastructure.detection_cache import (
        CachedDetections,
        StridedDetections,
    )
    from space_usage_heatmaps.evaluation.infrastructure.pipeline_track_source import (
        PipelineDetections,
        PipelineTrackSource,
        SelectedDetections,
    )
    from space_usage_heatmaps.perception.infrastructure import container

    cached = container.cache_params(config.detector, config.detector_params)

    def detection_at(stride):
        def detection_of(scene, camera):
            return container.build_detection(
                reader.video(scene, camera),
                config.detector,
                config.input_size,
                stride,
                data_dir() / "models",
                params=cached,
                engine=config.engine,
            )

        return detection_of

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

    def cache_at(stride):
        return CachedDetections(
            PipelineDetections(detection_at(stride)),
            data_dir() / "cache" / "detections" / detection_cache_key(config, stride),
            container.detection_metadata(
                config.detector, config.input_size, stride, cached, config.engine
            ),
            identity=video_identity,
        )

    detections = cache_at(1)
    if config.stride > 1:
        detections = StridedDetections(detections, cache_at(config.stride))
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
        for n, (config, source) in zip(names, runs, strict=True):
            tracks = source.tracks(scene, camera)
            if config.stitch is not None:  # the ids the floor scores, stitched
                tracks = stitched_tracks(reader.camera(scene, camera), tracks, config)
            per[n][key] = tracking_metrics(samples, tracks)
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
    scene_w = max(10, *(len(k) + 2 for k in t["scenes"][names[0]]))
    print(f"{'config':<{width}}{'scene':<{scene_w}}{'':<9}" + "".join(f"{m:>8}" for m in TRACKING))
    rows = [(n, t["scenes"][n], t["pooled"][n]) for n in names]
    if "delta" in t:
        rows.append(("delta", t["delta"]["scenes"], t["delta"]["pooled"]))
    for label, scenes, pool in rows:
        for where, m in [*scenes.items(), ("pooled", pool)]:
            cells = "".join(f"{m[k]:>8.3f}" for k in TRACKING)
            print(f"{label:<{width}}{where:<{scene_w}}{'tracking':<9}{cells}")
    if "delta" in t:
        d = {k: 100 * v for k, v in t["delta"]["pooled"].items()}
        holds = guard_holds(t)
        print(
            f"guard: HOTA {d['hota']:+.1f} points, IDF1 {d['idf1']:+.1f} points: "
            + ("holds" if holds else f"BROKEN (more than {GUARD_POINTS:g} point lower)")
        )


def _estimates(table):
    return {layer: {m: vars(e) for m, e in metrics.items()} for layer, metrics in table.items()}


def source_sha256():
    """sha256 of the package's Python sources: a stored report made by other
    code is not reused (space-usage eval --suite)."""
    h = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def eval_report_path(names, split, cell):
    return Path("reports") / f"eval-{'-vs-'.join(names)}-{split}-{cell}m.json"


def config_eval(config, compare, split, cell, resamples):
    """Score one config (and a second, paired) on a split from their tracks.
    (report dict, ConfigReport, names)."""
    reader = NvidiaMtmcReader(data_dir())
    names = [config_name(config)] + ([config_name(compare)] if compare else [])
    configs = [load_config(config)] + ([load_config(compare)] if compare else [])
    runs = [(c, track_source(c, reader)) for c in configs]
    r = evaluate_config(reader, SPLITS[split], runs, cell, n_resamples=resamples)
    tracking = score_tracking(reader, SPLITS[split], names, runs)
    tracking["guard_holds"] = guard_holds(tracking)
    n_blocks = {
        "scenes": {f"{run.scene}/{run.camera}": run.blocks for run in r.scenes},
        "pooled": sum(run.blocks for run in r.scenes),
    }
    report = {
        "dataset": r.dataset,
        "split": split,
        "cell_m": r.cell_m,
        "block_s": r.block_s,
        "n_resamples": r.n_resamples,
        "level": 0.95,
        "interval": INTERVAL,
        "pooled": "unweighted mean over scenes, blocks resampled within each scene",
        "source_sha256": source_sha256(),
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
    return report, r, names


def write_report(path, report):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path


def run_eval_config(args):
    if not args.split:
        print("space-usage: error: --config needs --split", file=sys.stderr)
        return 2
    report, r, names = config_eval(args.config, args.compare, args.split, args.cell, args.resamples)
    tracking = report["tracking"]

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

    path = write_report(eval_report_path(names, args.split, r.cell_m), report)
    print(f"\nreport {path}")
    if tracking["guard_holds"] is False:
        print("space-usage: tracking guard broken", file=sys.stderr)
        return GUARD_BROKEN
    return 0


# eval --cameras: the fused map (E2a) and the single-camera baseline it is compared to.
FUSION_CONFIG = "configs/e2a-fusion.yaml"
BASELINE_CONFIG = "configs/h5-best.yaml"
# Spec 7 targets as recalibrated after iteration 0 (iterations.md), dwell at 0.5 m.
TARGETS = {"mass_error": 0.10, "tvd": 0.30, "wape_zone": 0.30, "emd_m": 1.2}
FUSION_TITLES = {
    "full@joint": "every person, on the joint coverage of the fused cameras (E2a target)",
    "visible@joint": "people at least one fused camera sees, on the joint coverage",
    "full@anchor": "every person, on the baseline camera's coverage (E1 full@coverage)",
    "visible@anchor": "people the baseline camera sees, on its coverage (E1 visible@coverage)",
    "full@overlap": "every person, on cells two or more cameras cover (double counting)",
}


def _fusion_dict(f):
    return {"weighting": f.weighting, "power": _inf(f.power), "zone_m": f.zone_m}


def _inf(v):
    return "inf" if isinstance(v, float) and math.isinf(v) else v


def fused_export(scene, cameras, grid, fusion, config):
    """The fused map of a whole video as the viewer shows it: k-suppressed (R5)."""
    from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, suppress

    dwell, passage = suppress(grid.dwell, grid.passage, K_MIN_PEOPLE)
    spec = grid.spec
    return {
        "scene": scene,
        "cameras": list(cameras),
        "config": config,
        "fusion": _fusion_dict(fusion),
        "period": "the whole video",
        "k": K_MIN_PEOPLE,
        "cell_m": spec.cell_m,
        "origin": [spec.origin_x, spec.origin_y],
        "shape": [spec.ny, spec.nx],
        "dwell": dwell.round(3).tolist(),
        "passage": passage.round(3).tolist(),
    }


def _anchor_of(scene):
    for pairs in SPLITS.values():
        for s, camera in pairs:
            if s == scene:
                return camera
    raise ValueError(f"scene {scene} is in no split: no single-camera baseline camera")


def run_eval_fusion(args):
    from space_usage_heatmaps.evaluation.application.evaluate_fusion import evaluate_fusion

    config_path = args.config or FUSION_CONFIG
    compare_path = args.compare or BASELINE_CONFIG
    fusion = load_fusion(config_path)
    reader = NvidiaMtmcReader(data_dir())
    scenes = SPLITS[args.split] if args.split else ((args.scene, _anchor_of(args.scene)),)
    cameras = None if args.cameras == "all" else args.cameras.split(",")
    names = [config_name(config_path), config_name(compare_path)]
    configs = [load_config(config_path), load_config(compare_path)]
    runs = [(c, track_source(c, reader)) for c in configs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max(1, min(len(scenes), (os.cpu_count() or 2) // 2))) as pool:
        r = evaluate_fusion(
            reader,
            scenes,
            runs[0],
            runs[1],
            args.cell,
            fusion,
            args.resamples,
            cameras=cameras,
            map=pool.map,
        )
    where = f"split {args.split}" if args.split else f"scene {args.scene}"
    print(
        f"{r.dataset}  {where}  cell {r.cell_m} m  fusion {_fusion_dict(fusion)}  "
        f"paired block bootstrap: {r.block_s:g} s blocks, {r.n_resamples} resamples, 95 % CI"
    )
    print(f"fused = {names[0]}, single = {names[1]}; delta = fused - single, paired")
    for s in r.scenes:
        print(
            f"scene {s.scene}: cameras {', '.join(s.cameras)} fused ({len(s.cameras)}), "
            f"single-camera baseline {s.anchor}; joint coverage {s.joint_cells} cells, overlap "
            f"{s.overlap_cells}, baseline coverage {s.anchor_cells}; {s.samples} tracked boxes, "
            f"{s.dropped} outside grid, {s.blocks} blocks; unseen by every fused camera "
            f"{s.occlusion_joint:.3f} of the joint dwell, by the baseline camera "
            f"{s.occlusion_anchor:.3f} of its own"
        )
    rows = [SimpleNamespace(scene=s.scene, camera=s.anchor, blocks=s.blocks) for s in r.scenes]
    width = len("single") + 2
    for truth, res in r.results.items():
        print(f"\n{truth}: {FUSION_TITLES[truth]}")
        print(
            f"{'map':<{width}}{'scene':<10}{'layer':<9}{'blocks':>7}"
            + "".join(f"{m:>22}" for m in METRICS)
        )
        for i, label in enumerate(("fused", "single")):
            _rows(label, [sc[i] for sc in res.scenes], res.pooled[i], rows, width)
        _rows("delta", res.delta_scenes, res.delta_pooled, rows, width)
    if "full@overlap" in r.results:
        e = r.results["full@overlap"].pooled[0]["dwell"]["mass_error"]
        added = [s.sum_overlap_mass_error for s in r.scenes]
        print(
            "\noverlap mass error (dwell, every person, cells 2+ cameras cover): "
            f"{_ci(e, 0).strip()} (target <= 0.10: {'met' if e.value <= 0.10 else 'missed'}); "
            f"adding the cameras instead: {sum(added) / len(added):.3f}"
        )
    joint = r.results["full@joint"].pooled[0]["dwell"]
    targets = {
        m: {
            "target": t,
            "value": joint[m].value,
            "met": joint[m].value is not None and joint[m].value <= t,
        }
        for m, t in TARGETS.items()
    }
    print(
        "spec 7 targets on full@joint, fused dwell: "
        + ", ".join(
            f"{m} {v['value']:.3f} (<= {v['target']:g}: {'met' if v['met'] else 'missed'})"
            for m, v in targets.items()
        )
    )
    tag = f"-{args.split}-" if args.split else f"-s{args.scene}-"
    report = {
        "dataset": r.dataset,
        "split": args.split,
        "scene": args.scene,
        "cell_m": r.cell_m,
        "block_s": r.block_s,
        "n_resamples": r.n_resamples,
        "level": 0.95,
        "interval": INTERVAL,
        "source_sha256": source_sha256(),
        "fusion": _fusion_dict(fusion),
        "configs": {"fused": names[0], "single": names[1]},
        "scenes": [vars(s) | {"cameras": list(s.cameras)} for s in r.scenes],
        "targets": targets,
        "results": {
            truth: {
                "scenes": {
                    s.scene: {
                        "fused": _estimates(t[0]),
                        "single": _estimates(t[1]),
                        "delta": _estimates(d),
                    }
                    for s, t, d in zip(r.scenes, res.scenes, res.delta_scenes, strict=True)
                },
                "pooled": {
                    "fused": _estimates(res.pooled[0]),
                    "single": _estimates(res.pooled[1]),
                    "delta": _estimates(res.delta_pooled),
                },
            }
            for truth, res in r.results.items()
        },
    }
    path = write_report(
        Path("reports") / f"fusion-{'-vs-'.join(names)}{tag}{r.cell_m}m.json", report
    )
    for s in r.scenes:
        export = fused_export(s.scene, s.cameras, r.fused[s.scene], fusion, names[0])
        write_report(Path("reports") / f"fused-s{s.scene}-{names[0]}.json", export)
    print(f"\nreport {path}; fused maps reports/fused-s<scene>-{names[0]}.json (k={5})")
    return 0


# eval --cameras --metric mc-idf1: cross-camera ids on the floor (E2b).
ASSOCIATION_CONFIG = "configs/e2b-association.yaml"
# WILDTRACK's per-camera recipe: the E1 one at stride 1, as phase 8 scored it.
# Its frames are 2 fps; stride 2 (a GPU budget for ~150 NVIDIA videos) would
# detect once a second.
WILDTRACK_RECIPE = BASELINE_CONFIG
WILDTRACK_BLOCK_S = 10.0  # 400 frames at 2 fps: 20 blocks of 10 s (30 s would give 7)
# Pre-registered E2b target (reports/iterations-phase10.md, 2026-11-07, before
# any association ran): pooled mc-IDF1 >= baseline + 1/2 (oracle - baseline),
# .45 on NVIDIA (tune: .114 + .5 x .681); on WILDTRACK from its own baseline
# and oracle, never tuned there.
MC_IDF1_TARGET = 0.45
MC_IDF1_GAP_SHARE = 0.5
MC_IDF1_MIN_DELTA = 0.01  # spec 8 acceptance: CI excludes 0 and |delta| >= .01
IDENTITY_LABELS = ("associated", "baseline")


def _assoc_dict(a):
    return {f: getattr(a, f) for f in a.__dataclass_fields__}


def run_eval_identity(args):
    from concurrent.futures import ProcessPoolExecutor

    from space_usage_heatmaps.evaluation.application.evaluate_config import BLOCK_S
    from space_usage_heatmaps.evaluation.application.evaluate_identity import evaluate_identity
    from space_usage_heatmaps.evaluation.infrastructure.floor_identity import floor_identity
    from space_usage_heatmaps.occupancy.domain.association import NONE

    config_path = args.config or ASSOCIATION_CONFIG
    fusion, association = load_fusion(config_path), load_association(config_path)
    name = config_name(config_path)
    if args.dataset == "wildtrack":
        from space_usage_heatmaps.evaluation.infrastructure import wildtrack_reader

        reader = wildtrack_reader.WildtrackReader(data_dir())
        scenes, recipe_path, block_s = [wildtrack_reader.SCENE], WILDTRACK_RECIPE, WILDTRACK_BLOCK_S
        where, tag = "all cameras", "wildtrack"
    else:
        reader = NvidiaMtmcReader(data_dir())
        scenes = [s for s, _ in SPLITS[args.split]] if args.split else [args.scene]
        recipe_path, block_s = config_path, BLOCK_S
        where = f"split {args.split}" if args.split else f"scene {args.scene}"
        tag = args.split or f"s{args.scene}"
    config = load_config(recipe_path)
    source = track_source(config, reader)
    cameras = None if args.cameras == "all" else args.cameras.split(",")
    with ProcessPoolExecutor(max(1, min(len(scenes), (os.cpu_count() or 2) // 2))) as pool:
        r = evaluate_identity(
            reader,
            scenes,
            (config, source),
            fusion,
            dict(zip(IDENTITY_LABELS, (association, NONE), strict=True)),
            floor_identity,
            cell=args.cell,
            block_s=block_s,
            n_resamples=args.resamples,
            cameras=cameras,
            oracle=True,
            map=pool.map,
        )
    print(
        f"{r.dataset}  {where}  multi-camera IDF1 on the floor: TrackEval identity, floor "
        f"distance gate {r.gate_m} m; truth: people at least one camera sees, on the joint "
        f"coverage; paired block bootstrap: {r.block_s:g} s blocks, {r.n_resamples} "
        "resamples, 95 % CI"
    )
    print(
        f"association {_assoc_dict(association)}; per-camera recipe {config_name(recipe_path)}"
        "\nassociated = its global ids; baseline = every camera track its own id; oracle = "
        "camera tracks joined by their true person (a reference: it reads the truth)"
    )
    for s in r.scenes:
        ids = ", ".join(f"{k} {v}" for k, v in s.global_ids.items())
        print(
            f"scene {s.scene}: cameras {', '.join(s.cameras)} ({len(s.cameras)}); "
            f"{s.truth_people} people, {s.truth_points} truth points on {s.joint_cells} "
            f"joint cells; {s.camera_tracks} camera tracks; global ids: {ids}"
        )
    print(f"\n{'ids':<12}{'scene':<12}{'mc-IDF1':>24}{'IDP':>8}{'IDR':>8}")
    for i, label in enumerate(r.names):
        for k, s in enumerate(r.scenes):
            c = s.counts[label]
            print(
                f"{label:<12}{s.scene:<12}{_ci(r.idf1.scenes[k][i], 24)}"
                f"{c['idp']:>8.3f}{c['idr']:>8.3f}"
            )
        print(f"{label:<12}{'pooled':<12}{_ci(r.idf1.pooled[i], 24)}")
    for k, s in enumerate(r.scenes):
        print(f"{'delta':<12}{s.scene:<12}{_ci(r.idf1.delta_scenes[k], 24)}")
    d = r.idf1.delta_pooled
    print(f"{'delta':<12}{'pooled':<12}{_ci(d, 24)}  (delta associated - baseline, paired)")
    accepted = d.lo is not None and (d.lo > 0 or d.hi < 0) and abs(d.value) >= MC_IDF1_MIN_DELTA
    pooled = r.idf1.pooled[0].value
    target = MC_IDF1_TARGET
    if args.dataset == "wildtrack":
        base, oracle = r.idf1.pooled[1].value, r.idf1.pooled[2].value
        target = base + MC_IDF1_GAP_SHARE * (oracle - base)
    met = pooled >= target
    print(
        f"acceptance (CI excludes 0, |delta| >= {MC_IDF1_MIN_DELTA}): "
        + ("accepted" if accepted else "not accepted")
        + "; target "
        + f"mc-IDF1 >= {target:.3f}: {'met' if met else 'missed'}"
    )
    print(
        "\npassage side effect: distinct global ids per cell (associated) against the E2a "
        "fused map; both carry the E2a fused dwell"
    )
    rows = [SimpleNamespace(scene=s.scene, camera="joint", blocks=s.blocks) for s in r.scenes]
    for truth, res in r.passage.items():
        print(f"{truth}:")
        for i, label in enumerate(("global", "e2a")):
            _rows(label, [sc[i] for sc in res.scenes], res.pooled[i], rows, 8)
        _rows("delta", res.delta_scenes, res.delta_pooled, rows, 8)
    dwell_delta = [
        e.value
        for res in r.passage.values()
        for e in res.delta_pooled["dwell"].values()
        if e.value is not None
    ]
    print(f"dwell unchanged by ids: {all(v == 0 for v in dwell_delta)}")

    def est(e):
        return vars(e)

    report = {
        "dataset": r.dataset,
        "where": where,
        "metric": "mc-idf1",
        "gate_m": r.gate_m,
        "cell_m": r.cell_m,
        "block_s": r.block_s,
        "n_resamples": r.n_resamples,
        "level": 0.95,
        "interval": INTERVAL,
        "truth": "people at least one fused camera sees, on the joint coverage",
        "source_sha256": source_sha256(),
        "config": name,
        "recipe": config_name(recipe_path),
        "association": _assoc_dict(association),
        "fusion": _fusion_dict(fusion),
        "scenes": [vars(s) | {"cameras": list(s.cameras)} for s in r.scenes],
        "idf1": {
            "scenes": {
                s.scene: {label: est(r.idf1.scenes[k][i]) for i, label in enumerate(r.names)}
                | {"delta": est(r.idf1.delta_scenes[k])}
                for k, s in enumerate(r.scenes)
            },
            "pooled": {label: est(r.idf1.pooled[i]) for i, label in enumerate(r.names)}
            | {"delta": est(d)},
        },
        "acceptance": {"min_delta": MC_IDF1_MIN_DELTA, "accepted": bool(accepted)},
        "target": {"value": target, "met": bool(met)},
        "dwell_unchanged": all(v == 0 for v in dwell_delta),
        "passage": {
            truth: {
                "scenes": {
                    s.scene: {
                        "global": _estimates(t[0]),
                        "e2a": _estimates(t[1]),
                        "delta": _estimates(dt),
                    }
                    for s, t, dt in zip(r.scenes, res.scenes, res.delta_scenes, strict=True)
                },
                "pooled": {
                    "global": _estimates(res.pooled[0]),
                    "e2a": _estimates(res.pooled[1]),
                    "delta": _estimates(res.delta_pooled),
                },
            }
            for truth, res in r.passage.items()
        },
    }
    path = write_report(Path("reports") / f"mc-idf1-{name}-{tag}-{r.cell_m}m.json", report)
    for s in r.scenes:
        export = fused_export(s.scene, s.cameras, r.fused[s.scene], fusion, name)
        export |= {"passage_ids": "global", "association": _assoc_dict(association)}
        write_report(Path("reports") / f"fused-s{s.scene}-{name}.json", export)
    print(
        f"\nreport {path}; fused maps reports/fused-s<scene>-{name}.json (global-id "
        "passage: serve --fused refuses them, its passage layer is E2a)"
    )
    return 0


# External datasets (spec 5): evaluation only, never tuned on; each one's
# licence is in docs/licenses.md.
EXTERNAL = ("wildtrack", "caviar", "mot17")
MOT17_SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-09")  # the static cameras
GROUND_TARGET_M = 0.30  # spec 7: WILDTRACK median ground position error


def external_dataset(name):
    """(reader, [(scene, camera)]) of an external dataset under the data dir:
    WILDTRACK's most oblique camera, every CAVIAR corridor clip, MOT17's
    static-camera train sequences."""
    from space_usage_heatmaps.evaluation.infrastructure import (
        caviar_reader,
        mot17_reader,
        wildtrack_reader,
    )

    if name == "wildtrack":
        reader = wildtrack_reader.WildtrackReader(data_dir())
        scene = wildtrack_reader.SCENE
        return reader, [(scene, reader.most_oblique(scene))]
    if name == "caviar":
        reader = caviar_reader.CaviarReader(data_dir())
        return reader, [(clip, caviar_reader.CAMERA) for clip in reader.scenes()]
    reader = mot17_reader.Mot17Reader(data_dir())
    return reader, [(seq, mot17_reader.CAMERA) for seq in MOT17_SEQUENCES]


def wildtrack_ground_error(reader, scene, camera, config, source):
    """Floor points of the config's tracks (its foot point through the camera
    matrix) against the annotated floor positions, paired by box IoU."""
    import numpy as np

    from space_usage_heatmaps.evaluation.application.evaluate_oracle import camera_floor_points
    from space_usage_heatmaps.evaluation.infrastructure.ground_position import (
        ground_plane_error,
        matched_ground_error,
    )
    from space_usage_heatmaps.floorplan.domain.foot_point import camera_height, head_rule_applies

    t = source.tracks(scene, camera)
    cam = reader.camera(scene, camera)
    xs, ys, head = camera_floor_points(
        cam,
        t.boxes,
        config.foot_point,
        config.person_height_m,
        config.min_implied_height_m,
        t.track_ids,
        t.frames,
        config.track_heights(),
    )
    # Diagnostic split: boxes placed from their bottom edge or, judged hidden,
    # from their top edge at the track's height (auto, auto-track).
    groups = {"bottom-placed": ~head, "head-placed": head}
    samples = reader.samples(scene, camera)
    xy = np.column_stack([xs, ys])
    e = matched_ground_error(samples, t, xy, groups=groups)
    plane = ground_plane_error(samples, t, xy)
    return plane, {
        "camera": camera,
        "depression_deg": reader.depression_deg(scene, camera),
        "camera_height_m": camera_height(cam.projection),
        # auto-track's camera-height guard: whether its head rule may run here.
        "head_rule": (
            head_rule_applies(cam.projection, config.track_heights())
            if config.foot_point == "auto-track"
            else None
        ),
        "foot_point": config.foot_point,
        "head_boxes": int(head.sum()),
        **e,
        "target_m": GROUND_TARGET_M,
        "meets_target": e["median_m"] is not None and e["median_m"] <= GROUND_TARGET_M,
    }


def _ground_cells(g):
    def cm(v):
        return "n/a" if v is None else f"{100 * v:.1f} cm"

    def ratio(v):
        return "n/a" if v is None else f"{v:.2f}"

    return (
        f"median {cm(g['median_m'])}, p90 {cm(g['p90_m'])}, "
        f"recall {ratio(g['recall'])}, precision {ratio(g['precision'])} "
        f"({g['matched']} pairs)"
    )


def dataset_eval(dataset, config_path):
    """The report of a config on an external dataset, or None when its data is missing."""
    reader, scenes = external_dataset(dataset)
    if not scenes:
        return None
    name = config_name(config_path)
    config = load_config(config_path)
    stitch = None
    if config.stitch is not None:
        if reader.camera(*scenes[0]).world_to_image is None:
            reason = f"{reader.name} has no floor calibration: ids as the tracker gave them"
            stitch, config = {"applied": False, "reason": reason}, replace(config, stitch=None)
        else:
            stitch = {"applied": True, **config.stitch}
    source = track_source(config, reader)
    report = {
        "dataset": reader.name,
        "use": "evaluation only: never tuned on (spec 5); licence in docs/licenses.md",
        "config_name": name,
        "config": vars(config),
        "scenes": [f"{scene}/{camera}" for scene, camera in scenes],
    } | ({} if stitch is None else {"stitch": stitch})
    if dataset == "wildtrack":
        scene, camera = scenes[0]
        plane, g = wildtrack_ground_error(reader, scene, camera, config, source)
        report["ground_position"] = g
        report["ground_plane"] = plane
    report["tracking"] = score_tracking(reader, scenes, [name], [(config, source)])
    report["tracks_provenance"] = getattr(source, "provenance", None)
    return report


def run_eval_dataset(args):
    report = dataset_eval(args.dataset, args.config)
    if report is None:
        print(f"space-usage: error: no {args.dataset} data under {data_dir()}", file=sys.stderr)
        return 1
    name = report["config_name"]
    print(f"{report['dataset']}  config {name}  evaluation only, never tuned on")
    if "ground_position" in report:
        g, plane = report["ground_position"], report["ground_plane"]
        print(
            f"camera {g['camera']} (most oblique, {g['depression_deg']:.1f} degrees below the "
            f"horizon, {g['camera_height_m']:.2f} m high), foot point {g['foot_point']}"
            + ("" if g["head_rule"] is not False else " (head rule off: camera below heads)")
            + f", {g['truth_boxes']} annotated people, {g['tracked_boxes']} tracked boxes"
        )
        print("ground position error; it holds for the people each view pairs (recall):")
        print(f"  paired in the image, IoU >= {g['min_iou']}:  {_ground_cells(g)}")
        print(f"  paired on the floor, within {plane['gate_m']} m:  {_ground_cells(plane)}")
        verdict = "meets" if g["meets_target"] else "misses"
        print(
            f"target median <= {100 * GROUND_TARGET_M:.0f} cm (image pairing, "
            f"recall {'n/a' if g['recall'] is None else format(g['recall'], '.2f')}): {verdict}"
        )
        for group, m in g["groups"].items():
            if m["matched"]:
                print(
                    f"  {group}: {m['matched']} pairs, median {100 * m['median_m']:.1f} cm, "
                    f"p90 {100 * m['p90_m']:.1f} cm"
                )
    _print_tracking(report["tracking"], [name], max(len(name), len("config")) + 2)
    path = write_report(Path("reports") / f"external-{args.dataset}-{name}.json", report)
    print(f"\nreport {path}")
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


def bench_report_path(config):
    return Path("reports") / f"bench-{config_name(config)}.json"


def write_bench_report(config, minutes, result):
    """The bench's result, read back by eval --suite e1-final."""
    report = {
        "config": config_name(config),
        "minutes": minutes,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "video": f"scene {BENCH_SCENE} camera {BENCH_CAMERA}",
        **result,
    }
    return write_report(bench_report_path(config), report)


def run_bench(args):
    """End to end on scene 071's 1080p video, looped for --minutes."""
    config = load_config(args.config)
    reader = NvidiaMtmcReader(data_dir())
    cam = reader.camera(BENCH_SCENE, BENCH_CAMERA)
    spec = GridSpec.covering(*reader.scene_extent(BENCH_SCENE), cell_m=BENCH_CELL)
    if config.engine != "ultralytics":
        result = bench_pipeline(config, reader, cam, spec, args.minutes)
        print(json.dumps(result))
        write_bench_report(args.config, args.minutes, result)
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
    write_bench_report(args.config, args.minutes, result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="space-usage")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="score a heatmap source against ground truth")
    how = ev.add_mutually_exclusive_group()
    how.add_argument("--source", choices=SOURCES, help="oracle: ground truth as the prediction")
    how.add_argument("--config", help="run configuration, e.g. configs/iter0.yaml")
    how.add_argument(
        "--suite",
        choices=SUITES,
        help="e1-final: the E1 report of configs/h5-best.yaml, reports/e1-final.json and .md",
    )
    ev.add_argument(
        "--cameras",
        help=f"all, or ids separated by commas: fuse a scene's cameras (--scene or --split; "
        f"--config defaults to {FUSION_CONFIG}, --compare to {BASELINE_CONFIG})",
    )
    ev.add_argument("--scene", help="with --source or --cameras: scene id, e.g. 071")
    ev.add_argument("--camera", help="with --source: camera id, e.g. 645, or all")
    ev.add_argument(
        "--foot-point", choices=FOOT_POINTS, help="with --source gt-boxes: default bottom"
    )
    ev.add_argument(
        "--split", choices=sorted(SPLITS), help="with --config or --cameras: scenes to score"
    )
    ev.add_argument("--compare", help="with --config: second configuration, paired difference")
    ev.add_argument(
        "--dataset",
        choices=EXTERNAL,
        help="with --config: an external evaluation-only dataset instead of --split",
    )
    ev.add_argument(
        "--metric",
        choices=("heatmap", "mc-idf1"),
        default="heatmap",
        help=f"with --cameras: mc-idf1 scores cross-camera ids on the floor (E2b; --config "
        f"defaults to {ASSOCIATION_CONFIG}; --dataset wildtrack for WILDTRACK)",
    )
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
    se.add_argument(
        "--fused", help="a fused scene map of eval --cameras (reports/fused-s<scene>-<config>.json)"
    )
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
