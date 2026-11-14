"""Real-time budget, camera count (improvements plan, phase 3), as pre-registered
in docs/experiment-log.md, "Improvements plan, phase 3: real-time budget".

    uv run --no-sync python scripts/multicam-bench.py --stride 8 --cameras 4 --minutes 10

One process, one YOLOX-s 960 TensorRT fp16 detector shared by N camera videos
of scene 071, looped, served in turn (evaluation.infrastructure.round_robin).
Per camera and processed frame: decode (the skipped frames are grabbed), detect,
ByteTrack at 30 / stride fps, fill the skipped frames, auto-track floor points
and the 0.5 m dwell and passage grid. Stitching and fusion run per slot on the
recorded tracks and grids, not per frame, so they are not in the loop.

A camera count fits when the lowest per-camera p5 of processed fps is
>= 30 / stride (a 30 fps source kept up with). Writes
reports/multicam-bench-stride<S>-n<N>.json.
"""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pynvml

from space_usage_heatmaps.cli import BENCH_SCENE, data_dir, engine_label, load_config
from space_usage_heatmaps.evaluation.infrastructure.bench import process_vram_mb
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.evaluation.infrastructure.round_robin import round_robin
from space_usage_heatmaps.floorplan.domain.foot_point import camera_floor_points
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.perception.application.track_video import fill_stride, track_video
from space_usage_heatmaps.perception.infrastructure import container
from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource

RECIPE = "configs/e2a-stitch.yaml"
CELL_M = 0.5
FIRST = "645"  # the E1 camera of scene 071 first, then the others in id order


def camera_steps(source, detector, tracker, cam, config, spec, totals):
    """One next() per processed frame: detect, track, fill, floor points, grid."""
    stride = config.stride
    dwell = np.zeros(spec.ny * spec.nx)
    passage = set()
    heights = config.track_heights()
    for t in fill_stride(track_video(source, detector, tracker), stride):
        xs, ys, _ = camera_floor_points(
            cam,
            t.boxes,
            config.foot_point,
            config.person_height_m,
            config.min_implied_height_m,
            t.track_ids,
            np.full(len(t.track_ids), t.frame_index),
            heights,
        )
        col = np.floor((xs - spec.origin_x) / spec.cell_m).astype(int)
        row = np.floor((ys - spec.origin_y) / spec.cell_m).astype(int)
        inside = (col >= 0) & (col < spec.nx) & (row >= 0) & (row < spec.ny)
        cell = row[inside] * spec.nx + col[inside]
        np.add.at(dwell, cell, 1 / cam.fps)
        passage.update(zip(t.track_ids[inside].tolist(), cell.tolist(), strict=True))
        totals["dwell_s"] = float(dwell.sum())
        totals["passage_pairs"] = len(passage)
        if t.frame_index % stride == stride - 1:
            yield


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--cameras", type=int, required=True)
    p.add_argument("--minutes", type=float, default=10.0)
    args = p.parse_args()
    config = replace(load_config(RECIPE), stride=args.stride)
    reader = NvidiaMtmcReader(data_dir())
    ids = sorted((str(c) for c in reader.cameras(BENCH_SCENE)), key=lambda c: (c != FIRST, c))
    ids = ids[: args.cameras]
    if len(ids) != args.cameras:
        raise SystemExit(f"scene {BENCH_SCENE} has {len(ids)} cameras, not {args.cameras}")
    spec = GridSpec.covering(*reader.scene_extent(BENCH_SCENE), cell_m=CELL_M)
    detector = container.build_detector(
        config.detector,
        config.input_size,
        data_dir() / "models",
        config.detector_params,
        config.engine,
    )
    steps, totals = [], []
    for c in ids:
        cam = reader.camera(BENCH_SCENE, c)
        source = OpenCvFileSource(reader.video(BENCH_SCENE, c), config.stride, loop=True)
        tracker = container.build_tracker(
            config.tracker, source.fps / config.stride, cam.world_to_image, config.tracker_params
        )
        totals.append({})
        steps.append(camera_steps(source, detector, tracker, cam, config, spec, totals[-1]))
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        result = round_robin(steps, args.minutes * 60, vram_mb=lambda: process_vram_mb(handle))
    finally:
        pynvml.nvmlShutdown()
    providers = list(detector.providers)
    if providers != ["TensorrtExecutionProvider"]:
        raise SystemExit(f"the detector ran on {providers}, not TensorRT only")
    need = 30 / config.stride
    report = {
        "recipe": RECIPE,
        "stride": config.stride,
        "scene": BENCH_SCENE,
        "camera_ids": ids,
        "minutes": args.minutes,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine": engine_label(config.engine, providers, metadata.version("onnxruntime-gpu")),
        "needed_processed_fps": need,
        "fits": result["processed_fps_p5"] >= need,
        "source_fps_equivalent_mean": round(result["processed_fps_mean"] * config.stride, 2),
        "source_fps_equivalent_p5": round(result["processed_fps_p5"] * config.stride, 2),
        "vram_source": "NVML process usedGpuMemory, sampled once per round",
        "grids": totals,
        **result,
    }
    out = Path("reports") / f"multicam-bench-stride{config.stride}-n{args.cameras}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"stride {config.stride} N {args.cameras}: processed fps per camera mean "
        f"{result['processed_fps_mean']} p5 {result['processed_fps_p5']} (need {need:.2f}), "
        f"VRAM peak {result['vram_peak_mb']} MB, fits {report['fits']}; {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
