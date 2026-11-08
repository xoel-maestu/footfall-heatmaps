"""space-usage calibrate/run/serve --site: every camera of a site in one process (driving adapter).

A site config (YAML) names the cameras with their sources and calibrations,
the recipe config (detector, tracker, foot point), the stride, the floor grid
and the fusion. One TensorRT detector is shared by every camera, which take
turns (perception.infrastructure.round_robin_feed); each camera keeps its own
tracker and floor points (occupancy.application.record_cameras). The fused
grid is written per 15-minute slot under the site id, with k=5 applied after
fusion (occupancy.application.record_site).

The fusion weights are frozen per site: `space-usage calibrate --site` runs
the cameras over the calibration window and writes them once to the weights
file the site config names; recording reads that file and refuses weights
made for other cameras, another grid or another fusion. They come from the
cameras' own recorded dwell and coverage, never from ground truth.

Relative source and weights paths are under the data dir
(SPACE_USAGE_DATA_DIR); the recipe path is as given, else next to the site file.
"""

import collections
import contextlib
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from space_usage_heatmaps.cli import (
    calibration_of,
    data_dir,
    load_config,
    utc_epoch,
)
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.application.calibrate_site import CalibrateSite
from space_usage_heatmaps.occupancy.application.purge_slots import RETENTION_DAYS, purge_slots
from space_usage_heatmaps.occupancy.application.record_cameras import CameraPipe, record_cameras
from space_usage_heatmaps.occupancy.application.record_site import RecordSite
from space_usage_heatmaps.occupancy.domain.fusion import Fusion
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
from space_usage_heatmaps.perception.application.reconnect import reconnecting_frames
from space_usage_heatmaps.perception.domain.privacy_mask import PrivacyMask
from space_usage_heatmaps.perception.infrastructure.round_robin_feed import round_robin_feed

SITE_KEYS = {
    "site", "recipe", "stride", "extent_m", "cell_m", "retention_days", "start_utc",
    "fusion", "calibration", "cameras",
}  # fmt: skip
CAMERA_KEYS = {"id", "source", "calibration", "masks"}
FUSION_KEYS = {"weighting", "power", "zone_m", "covered_cells", "weights"}
WARM_UP_S = 5  # stats: the first seconds (first inferences) are left out of the rates
RECENT_FRAMES = 30  # the viewer's per-camera rate: over a camera's last frames
WINDOW_S = 10  # stats: a live source caps a camera at its own rate (30 / stride), so per-second
# counts jitter around it by the bin edges; 10 s windows show whether it keeps up
STALLED_S = 5.0  # the viewer: a camera with no frame for this long reads as stalled


@dataclass(frozen=True)
class SiteCamera:
    id: str
    source: str  # a file path or rtsp:// URL, resolved
    calibration: str  # nvidia-mtmc/<scene>/<camera>
    masks: dict = field(default_factory=dict)

    @property
    def stream(self):
        return self.source.startswith("rtsp://")


@dataclass(frozen=True)
class Site:
    id: str
    config: object  # RunConfig of the recipe, at the site's stride
    spec: GridSpec
    retention_days: float
    start_utc: object
    fusion: Fusion
    weights_path: Path
    calibration_s: float
    cameras: tuple

    @property
    def camera_ids(self):
        return [c.id for c in self.cameras]

    @property
    def streams(self):
        return self.cameras[0].stream


def _under_data(p):
    p = str(p)
    return p if p.startswith("rtsp://") or Path(p).is_absolute() else str(data_dir() / p)


def _unknown(d, allowed, where):
    if unknown := sorted(set(d) - allowed):
        raise ValueError(f"{where}: unknown keys {unknown}")


def load_site(path):
    path = Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    _unknown(d, SITE_KEYS, f"site {path}")
    for key in ("site", "recipe", "stride", "extent_m", "fusion", "cameras"):
        if key not in d:
            raise ValueError(f"site {path}: no {key}")
    recipe = Path(d["recipe"])
    if not recipe.is_absolute() and not recipe.exists():
        recipe = path.parent / recipe
    config = load_config(recipe)
    if config.stitch is not None:
        raise ValueError(
            f"recipe {recipe}: stitch joins tracks across later frames, an evaluation step; "
            "recording keeps the tracker's own ids"
        )
    if not (isinstance(d["stride"], int) and d["stride"] >= 1):
        raise ValueError(f"site {path}: stride must be a positive integer")
    config = replace(config, stride=d["stride"])
    extent = d["extent_m"]
    if not isinstance(extent, list) or len(extent) != 4:
        raise ValueError(f"site {path}: extent_m must be [xmin, ymin, xmax, ymax]")
    fusion = dict(d["fusion"] or {})
    _unknown(fusion, FUSION_KEYS, f"site {path} fusion")
    if "weights" not in fusion:
        raise ValueError(f"site {path}: fusion.weights names the frozen weights file")
    weights = fusion.pop("weights")
    cameras = []
    for c in d["cameras"] or []:
        _unknown(c, CAMERA_KEYS, f"site {path} camera")
        masks = c.get("masks") or {}
        _unknown(masks, {"image", "floor"}, f"site {path} camera masks")
        cameras.append(SiteCamera(str(c["id"]), _under_data(c["source"]), c["calibration"], masks))
    if not cameras:
        raise ValueError(f"site {path}: no cameras")
    if len({c.id for c in cameras}) != len(cameras):
        raise ValueError(f"site {path}: camera ids repeat")
    if len({c.stream for c in cameras}) != 1:
        raise ValueError(f"site {path}: the sources must be all files or all rtsp:// streams")
    retention = d.get("retention_days", RETENTION_DAYS)
    if not retention > 0:
        raise ValueError(f"site {path}: retention_days must be positive")
    minutes = (d.get("calibration") or {}).get("minutes", 15)
    return Site(
        id=str(d["site"]),
        config=config,
        spec=GridSpec.covering(*extent, cell_m=d.get("cell_m", 0.5)),
        retention_days=retention,
        start_utc=d.get("start_utc"),
        fusion=Fusion(**fusion),
        weights_path=Path(_under_data(weights)),
        calibration_s=float(minutes) * 60,
        cameras=tuple(cameras),
    )


def _fusion_json(f):
    return json.dumps(
        {"weighting": f.weighting, "power": f.power, "zone_m": f.zone_m,
         "covered_cells": f.covered_cells}, sort_keys=True,
    )  # fmt: skip


def _spec_array(spec):
    return np.array([spec.origin_x, spec.origin_y, spec.cell_m, spec.nx, spec.ny], float)


def load_weights(site):
    """The site's frozen weights, checked against its cameras, grid and fusion."""
    if not site.weights_path.exists():
        raise ValueError(
            f"site {site.id}: no fusion weights at {site.weights_path}; "
            "run space-usage calibrate --site first"
        )
    with np.load(site.weights_path, allow_pickle=False) as f:
        cameras, weights = [str(c) for c in f["cameras"]], f["weights"]
        spec, fusion = f["spec"], str(f["fusion"])
    if cameras != site.camera_ids:
        raise ValueError(
            f"site {site.id}: the weights were calibrated for cameras {cameras}, "
            f"the site has {site.camera_ids}; calibrate again"
        )
    if not np.allclose(spec, _spec_array(site.spec)):
        raise ValueError(f"site {site.id}: the weights are on another floor grid; calibrate again")
    if json.loads(fusion) != json.loads(_fusion_json(site.fusion)):
        raise ValueError(
            f"site {site.id}: the weights were made with fusion {fusion}; calibrate again"
        )
    return weights


class CameraMeter:
    """Processed frames per camera: per-second counts for the stats file, the
    last frames' times for the viewer. Counts and times only."""

    def __init__(self, cameras, clock=time.monotonic):
        self.clock = clock
        self.cameras = list(cameras)
        self.start = None
        self.ended = False
        self.recording = False
        self.frames = dict.fromkeys(self.cameras, 0)
        self.bins = {c: collections.Counter() for c in self.cameras}
        self.recent = {c: collections.deque(maxlen=RECENT_FRAMES) for c in self.cameras}
        self.feed_stats = {}

    def tick(self, camera_id):
        now = self.clock()
        if self.start is None:
            self.start = now
        self.frames[camera_id] += 1
        self.bins[camera_id][int(now - self.start)] += 1
        self.recent[camera_id].append(now)

    def status(self):
        now = self.clock()
        out = []
        for c in self.cameras:
            r = self.recent[c]
            if not self.recording:
                state = "not recording"
            elif self.ended:
                state = "ended"
            elif not r:
                state = "starting"
            elif now - r[-1] > STALLED_S:
                state = "stalled"
            else:
                state = "recording"
            fps = (len(r) - 1) / (r[-1] - r[0]) if len(r) > 1 and r[-1] > r[0] else None
            s = self.feed_stats.get(c, {})
            out.append(
                {
                    "id": c,
                    "state": state,
                    "processed_fps": None if fps is None else round(fps, 2),
                    "processed_frames": self.frames[c],
                    "frames_skipped": s.get("dropped", 0),
                    "source_drops": s.get("drops", 0),
                }
            )
        return out

    def report(self):
        """Per camera: frames, mean and p5 of the per-second processed rate after
        the warm-up, whole seconds only."""
        last = int(self.clock() - self.start) if self.start is not None else 0
        seconds = range(WARM_UP_S, last)  # the last second is partial
        cameras = []
        for c in self.cameras:
            rates = np.array([self.bins[c].get(s, 0) for s in seconds], float)
            s = self.feed_stats.get(c, {})
            cameras.append(
                {
                    "camera": c,
                    "processed_frames": self.frames[c],
                    "processed_fps_mean": round(float(rates.mean()), 2) if len(rates) else None,
                    "processed_fps_p5": round(float(np.percentile(rates, 5)), 2)
                    if len(rates)
                    else None,
                    "processed_fps_p5_10s": _p5_windows(rates, WINDOW_S),
                    "frames_skipped": s.get("dropped", 0),
                    "source_drops": s.get("drops", 0),
                }
            )
        p5 = [c["processed_fps_p5"] for c in cameras if c["processed_fps_p5"] is not None]
        return {
            "timed_seconds": len(seconds),
            "warm_up_s": WARM_UP_S,
            "processed_fps_p5_min": min(p5) if p5 else None,
            "cameras": cameras,
        }


def _p5_windows(rates, n):
    """5th percentile of the rate over consecutive whole windows of n seconds."""
    windows = len(rates) // n
    if not windows:
        return None
    means = np.asarray(rates[: windows * n]).reshape(windows, n).mean(axis=1)
    return round(float(np.percentile(means, 5)), 2)


def _inputs(site, stop):
    """(detector, camera id -> CameraPipe, camera id -> make(on_drop), coverage)."""
    from space_usage_heatmaps.perception.infrastructure import container
    from space_usage_heatmaps.perception.infrastructure.opencv_file_source import (
        OpenCvFileSource,
    )
    from space_usage_heatmaps.perception.infrastructure.opencv_stream_source import (
        OpenCvStreamSource,
    )

    config = site.config
    detector = container.build_detector(
        config.detector,
        config.input_size,
        data_dir() / "models",
        config.detector_params,
        config.engine,
    )
    start = site.start_utc
    t0 = time.time() if start is None else utc_epoch(start)
    pipes, sources, coverage = {}, {}, []
    for c in site.cameras:
        cam = calibration_of(c.calibration)
        coverage.append(Homography(cam.world_to_image).sees(*site.spec.cell_centres(),
                                                             cam.width, cam.height))  # fmt: skip
        mask = PrivacyMask(c.masks.get("image") or (), c.masks.get("floor") or (),
                           cam.world_to_image)  # fmt: skip
        if c.stream:
            first = [OpenCvStreamSource(c.source, config.stride)]
            fps = first[0].fps

            def open_stream(first=first, url=c.source):
                return first.pop() if first else OpenCvStreamSource(url, config.stride)

            def make(on_drop, open_stream=open_stream):
                return reconnecting_frames(open_stream, stop.wait, on_drop)

            def time_of(frame):
                return time.time()
        else:
            fps = OpenCvFileSource(c.source, config.stride).fps

            def make(on_drop, path=c.source):
                return OpenCvFileSource(path, config.stride).frames()

            def time_of(frame, t0=t0):
                return t0 + frame.timestamp

        tracker = container.build_tracker(
            config.tracker, fps / config.stride, cam.world_to_image, config.tracker_params
        )
        pipes[c.id] = CameraPipe(mask, tracker, cam, time_of, config.stride / fps)
        sources[c.id] = make
    return detector, pipes, sources, np.stack(coverage)


def _drive(site, recorder, stop, meter, until=None):
    detector, pipes, sources, coverage = _inputs(site, stop)
    if callable(recorder):
        recorder = recorder(coverage)
    meter.recording = True
    feed = round_robin_feed(sources, paced=not site.streams, stop=stop, stats=meter.feed_stats)
    try:
        n = record_cameras(
            feed, detector, pipes, site.config, recorder, until=until, on_frame=meter.tick
        )
    finally:
        stopped = stop.is_set()
        stop.set()  # a stream reader waiting to reconnect wakes up and leaves
        feed.close()
        meter.ended = True
    if not stopped:
        stop.clear()  # the sources ended on their own: serve tells "ended" from "stopped"
    return n, recorder


@contextlib.contextmanager
def _stop_on_signals(stop):
    """Ctrl+C and SIGTERM set stop, so the open slots are still written; the
    previous handlers come back afterwards. Only the main thread installs them."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        stop.set()

    old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def run_calibrate(args):
    """space-usage calibrate --site: the frozen fusion weights of a site."""
    site = load_site(args.site)
    if site.weights_path.exists():
        raise ValueError(
            f"site {site.id}: weights {site.weights_path} already exist and are frozen; "
            "delete the file to calibrate again"
        )
    stop = threading.Event()
    meter = CameraMeter(site.camera_ids)
    holder = {}

    def calibration(coverage):
        holder["c"] = CalibrateSite(
            site.camera_ids, coverage, site.spec, site.fusion, site.calibration_s
        )
        return holder["c"]

    with _stop_on_signals(stop):
        _, cal = _drive(site, calibration, stop, meter, until=lambda: holder["c"].complete)
    weights = cal.weights()
    site.weights_path.parent.mkdir(parents=True, exist_ok=True)
    window = {
        "start_utc": cal.start,
        "seconds": site.calibration_s,
        "complete": cal.complete,
        "dwell_s": cal.dwell_s(),
    }
    tmp = site.weights_path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        cameras=np.array(site.camera_ids),
        weights=weights,
        spec=_spec_array(site.spec),
        fusion=np.array(_fusion_json(site.fusion)),
        window=np.array(json.dumps(window, sort_keys=True)),
    )
    tmp.replace(site.weights_path)
    joint = int((weights.sum(0) > 0).sum())
    print(
        f"site {site.id}: weights of {len(site.cameras)} cameras on {joint} covered cells "
        f"from a {site.calibration_s / 60:g}-min window "
        f"({'complete' if cal.complete else 'sources ended first'}) in {site.weights_path}"
    )
    for c, s in cal.dwell_s().items():
        print(f"  camera {c}: {s:.0f} person-seconds in the window")
    return 0


def run_record_site(args, live=None, stop=None, meter=None):
    """space-usage run --site: every camera into the site's fused 15-min slots.
    live: a LiveSource (space-usage serve --record); stop: an Event that ends it."""
    site = load_site(args.site)
    weights = load_weights(site)
    stop = stop or threading.Event()
    meter = meter or CameraMeter(site.camera_ids)
    store = SqliteSlotStore(args.db)
    recorder = RecordSite(
        store, site.id, site.camera_ids, weights, site.spec, site.retention_days, time.time
    )
    inner = recorder
    if live is not None:
        recorder = live.attach(recorder)
    purge_slots(store, time.time(), site.retention_days)
    try:
        with _stop_on_signals(stop):
            n, _ = _drive(site, recorder, stop, meter)
    finally:
        store.close()
    stats = meter.report()
    print(
        f"site {site.id}: {len(site.cameras)} cameras, {n} frames, {inner.written} slots "
        f"({site.spec.ny}x{site.spec.nx} cells of {site.spec.cell_m} m) in {args.db}; "
        f"lowest per-camera p5 {stats['processed_fps_p5_min']} processed fps"
    )
    if getattr(args, "stats", None):
        report = {"site": site.id, "stride": site.config.stride, "frames": n} | stats
        Path(args.stats).write_text(json.dumps(report, indent=2) + "\n")
    if inner.past_retention:
        print(
            f"site {site.id}: {inner.past_retention} slots past retention "
            f"({site.retention_days} days) were not written; check start_utc",
            file=sys.stderr,
        )
        return 1
    return 0


def site_info(site, meter):
    """GET /api/site: the site, its fusion and each camera's state. No per-track data."""

    def info():
        f = site.fusion
        return {
            "site": site.id,
            "fusion": {
                "weighting": f.weighting,
                "power": f.power,
                "zone_m": f.zone_m,
                "covered_cells": f.covered_cells,
            },  # fmt: skip
            "stride": site.config.stride,
            "cameras": meter.status(),
        }

    return info
