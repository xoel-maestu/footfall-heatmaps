"""One global recall factor for single-camera mean occupancy (improvements plan,
phase 7, H7), as pre-registered in iterations.md, "Improvements plan, phase 7"
(row H7).

    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/recall-factor.py tune
    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/recall-factor.py sealed   # once
    (refuses if reports/recall-factor-sealed.json or .started exists)

Hypothesis: a single camera's mean occupancy is biased low by a stable recall
factor. f = sum of visible truth / sum of shown mean occupancy over every
calibrated camera of the tune scenes (071-075), applied to single-camera
readings only, never to sites. Score: the median |error| of f x the shown mean
occupancy against the visible truth, over every calibrated camera of the
sealed scenes (076-080), scored once. Target and acceptance: <= .10. Also
reported: the p90, the uncorrected median and p90, and the tune per-camera
error distribution (raw and corrected), to show whether one factor is plausible.

Choices fixed here before the first run (not in the pre-registration):
- Recipe: configs/e2a-record.yaml (what `space-usage run` records with) at
  stride 10, replayed from the stride-2 TensorRT detection cache
  (track_source(..., detections_stride=2)): the tracker runs on every 10th
  frame, CPU only. Units: every calibrated camera of the split
  (reader.cameras), each read as a single camera.
- Shown mean occupancy, as the product computes it: only the tracker's own
  rows on processed frames (frame % 10 == 0; fill_stride's filled rows are an
  evaluation device the recorder never sees), their floor points through
  camera_floor_points frame by frame (auto-track is causal), into RecordSlots
  with an in-memory SqliteSlotStore and k=5, every processed frame of the
  video included (empty ones add observed time), each worth stride / fps, the
  video's frame 0 at 2026-11-06T09:00:00Z, the slot closed (incomplete) at
  the video's end. Then read_minutes and slot_reading: the whole grid, 0 for a
  minute below 5 ids, over the observed seconds stored. Grid: the scene's
  extent (GridSpec.covering(reader.scene_extent, 0.5)).
- A camera's reading over several slots (a video longer than one quarter
  after 09:00) is pooled: sum of person-seconds over sum of observed seconds.
  The videos are about 13 min, so a camera has one slot.
- Visible truth, as scripts/occupancy-truth.py for a single camera: the
  ground-truth rows this camera has a visible box of, on its coverage
  (Homography.sees on the cell centres), inside the slot window, binned by
  block_layers, over the same observed seconds.
- Error: f x shown / truth - 1 on exact values (the decision). The error of
  the value as the page rounds it (one decimal) is reported beside it. A
  camera whose truth is 0 has no relative error: left out and counted.
- f is computed from the tune report and frozen into reports/recall-factor-tune.json
  before the sealed run; the sealed run refuses to run twice.
"""

import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

RECIPE = "configs/e2a-record.yaml"
STRIDE, CACHED_STRIDE = 10, 2
CELL_M = 0.5
T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
TARGET = 0.10
END_OF_TIME = 4_102_444_800
REPORTS = Path("reports")


def _truth_module():
    path = Path(__file__).resolve().parent / "occupancy-truth.py"
    spec = importlib.util.spec_from_file_location("occupancy_truth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(spec, tracks, floor_of, frame_count, stride, fps, t0):
    """The slots the recorder writes for these tracks, read back as the page
    reads them: [{slot_start_utc, person_s, observed_s, mean_occupancy}].
    floor_of(boxes, ids, frames) -> (xs, ys) of one processed frame's rows."""
    from space_usage_heatmaps.occupancy.application.read_slots import read_minutes
    from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
    from space_usage_heatmaps.occupancy.domain.slots import slot_reading
    from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

    frames = np.asarray(tracks.frames)
    keep = frames % stride == 0
    order = np.argsort(frames[keep], kind="stable")
    ids = np.asarray(tracks.track_ids)[keep][order]
    fr = frames[keep][order]
    boxes = np.asarray(tracks.boxes, float).reshape(-1, 4)[keep][order]
    store = SqliteSlotStore(":memory:")
    recorder = RecordSlots(store, "cam", spec)
    seconds = stride / fps
    for f in range(0, frame_count, stride):
        a, b = np.searchsorted(fr, f), np.searchsorted(fr, f, side="right")
        xs, ys = floor_of(boxes[a:b], ids[a:b], fr[a:b]) if b > a else ([], [])
        recorder.add(t0 + f / fps, ids[a:b], np.asarray(xs), np.asarray(ys), seconds)
    recorder.close()
    rows = []
    for m in read_minutes(store, "cam", 0, END_OF_TIME):
        r = slot_reading(None, m, m.incomplete)
        rows.append({
            "slot_start_utc": m.start_utc,
            "person_s": float(np.sum(m.person_s)),
            "observed_s": r["observed_s"],
            "mean_occupancy": r["mean_occupancy"],
        })  # fmt: skip
    store.close()
    return rows


def fit_factor(cams):
    """f = sum of truth / sum of shown mean occupancy."""
    return sum(c["truth"] for c in cams) / sum(c["shown"] for c in cams)


def error_summary(cams, f, on_page=None):
    """Relative errors of f x shown against truth; cameras without truth left out."""
    used = [c for c in cams if c["truth"] > 0]
    value = (lambda x: on_page(x)) if on_page else (lambda x: x)
    errors = [value(f * c["shown"]) / c["truth"] - 1 for c in used]
    a = np.abs(errors)
    return {
        "n": len(used),
        "left_out": len(cams) - len(used),
        "errors": errors,
        "median_abs": float(np.median(a)) if len(a) else None,
        "p90_abs": float(np.percentile(a, 90)) if len(a) else None,
        "median": float(np.median(errors)) if len(a) else None,
        "min": float(min(errors)) if errors else None,
        "max": float(max(errors)) if errors else None,
        "p10": float(np.percentile(errors, 10)) if errors else None,
        "p90": float(np.percentile(errors, 90)) if errors else None,
    }


def measure_scene(scene):
    import cv2

    from space_usage_heatmaps.cli import data_dir, load_config, track_source
    from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import (
        NvidiaMtmcReader,
    )
    from space_usage_heatmaps.floorplan.domain.foot_point import camera_floor_points
    from space_usage_heatmaps.floorplan.domain.homography import Homography
    from space_usage_heatmaps.occupancy.domain.grid import GridSpec

    truth_person_seconds = _truth_module().truth_person_seconds
    reader = NvidiaMtmcReader(data_dir())
    config = replace(load_config(RECIPE), stride=STRIDE)
    source = track_source(config, reader, detections_stride=CACHED_STRIDE)
    spec = GridSpec.covering(*reader.scene_extent(scene), cell_m=CELL_M)
    out = []
    for c in reader.cameras(scene):
        t_start = time.time()
        cam = reader.camera(scene, c)
        cap = cv2.VideoCapture(str(reader.video(scene, c)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        heights = config.track_heights()

        def floor_of(boxes, ids, frames, cam=cam, heights=heights):
            xs, ys, _ = camera_floor_points(cam, boxes, config.foot_point,
                                            config.person_height_m, config.min_implied_height_m,
                                            ids, frames, heights)  # fmt: skip
            return xs, ys

        slots = replay(spec, source.tracks(scene, c), floor_of, frame_count, STRIDE, cam.fps, T0)
        coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
        truth = reader.floor_truth(scene, c)
        v = truth.visible
        truth_s = [
            truth_person_seconds(
                spec, truth.frames[v], truth.world[v], cam.fps, coverage, T0, s["slot_start_utc"]
            )  # fmt: skip
            for s in slots
        ]
        observed = sum(s["observed_s"] or 0 for s in slots)
        person_s = sum(s["person_s"] for s in slots)
        out.append({
            "scene": scene,
            "camera": c,
            "frames": frame_count,
            "fps": cam.fps,
            "coverage_cells": int(coverage.sum()),
            "slots": [s | {"truth_person_s": t} for s, t in zip(slots, truth_s, strict=True)],
            "observed_s": observed,
            "shown": person_s / observed if observed else None,
            "truth": sum(truth_s) / observed if observed else None,
            "seconds": round(time.time() - t_start, 1),
        })  # fmt: skip
        print(f"{scene}/{c}: shown {out[-1]['shown']}, truth {out[-1]['truth']}", flush=True)
    return out


def measure(split):
    from space_usage_heatmaps.cli import source_sha256
    from space_usage_heatmaps.evaluation.domain.splits import SPLITS

    scenes = [s for s, _ in SPLITS[split]]
    with ProcessPoolExecutor(len(scenes)) as pool:
        cams = [c for scene in pool.map(measure_scene, scenes) for c in scene]
    missing = [f"{c['scene']}/{c['camera']}" for c in cams if c["shown"] is None]
    cams = [c for c in cams if c["shown"] is not None]
    return {
        "hypothesis": "H7: one global recall factor for single-camera mean occupancy",
        "split": split,
        "scenes": scenes,
        "recipe": f"{RECIPE} at stride {STRIDE} from the stride-{CACHED_STRIDE} detection cache",
        "truth": "visible to the camera, on its coverage, over the observed seconds stored",
        "source_sha256": source_sha256(),
        "cameras_without_observed_time": missing,
        "cameras": cams,
    }


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    on_page = _truth_module().on_page
    mode = (sys.argv[1:] or [None])[0]
    t0 = time.time()
    if mode == "tune":
        report = measure("tune")
        f = fit_factor(report["cameras"])
        report |= {
            "factor": f,
            "raw": error_summary(report["cameras"], 1.0),
            "corrected": error_summary(report["cameras"], f),
            "per_scene_factor": {
                s: fit_factor([c for c in report["cameras"] if c["scene"] == s])
                for s in report["scenes"]
            },
            "seconds": round(time.time() - t0),
        }
        write(REPORTS / "recall-factor-tune.json", report)
        r, k = report["raw"], report["corrected"]
        print(f"f = {f:.4f}; tune raw median |err| {r['median_abs']:.3f} (range "
              f"{r['min']:+.3f} to {r['max']:+.3f}); corrected median |err| {k['median_abs']:.3f}, "
              f"p90 {k['p90_abs']:.3f}; per scene f {report['per_scene_factor']}")  # fmt: skip
    elif mode == "sealed":
        out = REPORTS / "recall-factor-sealed.json"
        started = REPORTS / "recall-factor-sealed.started"
        for p in (out, started):
            if p.exists():
                raise SystemExit(f"{p} exists: the sealed split is scored once")
        tune = json.loads((REPORTS / "recall-factor-tune.json").read_text())
        f = tune["factor"]
        # written before any scoring, so a run that crashes halfway still counts
        started.parent.mkdir(parents=True, exist_ok=True)
        started.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        report = measure("sealed")
        corrected = error_summary(report["cameras"], f)
        report |= {
            "factor": f,
            "factor_from": "reports/recall-factor-tune.json",
            "corrected": corrected,
            "uncorrected": error_summary(report["cameras"], 1.0),
            "corrected_on_page": error_summary(report["cameras"], f, on_page),
            "target_median_abs": TARGET,
            "accepted": corrected["median_abs"] <= TARGET,
            "seconds": round(time.time() - t0),
        }
        write(out, report)
        u = report["uncorrected"]
        print(f"sealed, f = {f:.4f}: corrected median |err| {corrected['median_abs']:.3f}, "
              f"p90 {corrected['p90_abs']:.3f}; uncorrected median {u['median_abs']:.3f}, "
              f"p90 {u['p90_abs']:.3f}; accepted {report['accepted']}")  # fmt: skip
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
