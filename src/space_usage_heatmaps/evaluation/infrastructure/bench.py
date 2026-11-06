"""Sustained end-to-end speed: decode -> detect -> track -> floor grid, for any pipeline.

fps counts source-video frames per wall-clock second (with stride N, the
pipeline fills the skipped frames back, so tracked yields every source frame),
from the first frame on (model load and the first inference are excluded).
fps_p5 is the 5th percentile of the per-second rates. VRAM is this process's
GPU memory as the driver reports it (NVML), the peak of one sample every 30
frames: it includes the CUDA context and every runtime's cached blocks, and is
what another process on the 4 GB card would find taken.
"""

import os
import time

import numpy as np
import pynvml

from space_usage_heatmaps.evaluation.application.evaluate_oracle import camera_floor_points

VRAM_EVERY = 30  # frames between two NVML samples


def process_vram_mb(handle):
    used = [
        p.usedGpuMemory
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if p.pid == os.getpid() and p.usedGpuMemory is not None
    ]
    return sum(used) / 2**20


def bench(tracked, config, cam, spec, minutes):
    """tracked: iterator of (frame index, track ids, x1 y1 x2 y2 boxes), started
    lazily (the model may load on the first next())."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return _run(tracked, config, cam, spec, minutes, handle)
    finally:
        pynvml.nvmlShutdown()


def _run(tracked, config, cam, spec, minutes, handle):
    fps = cam.fps
    dwell = np.zeros(spec.ny * spec.nx)
    passage = set()  # (track id, cell)
    rates, peak, frames = [], 0.0, 0
    start = window = None
    in_window = 0
    heights = config.track_heights()  # auto-track: each track's height so far, the scored rule
    for i, ids, boxes in tracked:
        xs, ys, _ = camera_floor_points(
            cam,
            boxes,
            config.foot_point,
            config.person_height_m,
            config.min_implied_height_m,
            ids,
            np.full(len(ids), i),
            heights,
        )
        col = np.floor((xs - spec.origin_x) / spec.cell_m).astype(int)
        row = np.floor((ys - spec.origin_y) / spec.cell_m).astype(int)
        inside = (col >= 0) & (col < spec.nx) & (row >= 0) & (row < spec.ny)
        cell = row[inside] * spec.nx + col[inside]
        np.add.at(dwell, cell, 1 / fps)
        passage.update(zip(ids[inside].tolist(), cell.tolist(), strict=True))

        now = time.monotonic()
        if start is None:  # timing starts after the first frame
            start = window = now
            peak = process_vram_mb(handle)
            continue
        frames += 1
        in_window += 1
        if now - window >= 1.0:
            rates.append(in_window / (now - window))
            window, in_window = now, 0
        if frames % VRAM_EVERY == 0:
            peak = max(peak, process_vram_mb(handle))
        if now - start >= minutes * 60:
            break
    elapsed = time.monotonic() - start
    peak = max(peak, process_vram_mb(handle))
    return {
        "fps_mean": round(frames / elapsed, 2),
        "fps_p5": round(float(np.percentile(rates, 5)), 2) if rates else None,
        "vram_peak_mb": round(peak, 1),
        "frames": frames,
        "seconds": round(elapsed, 1),
        "vram_source": "NVML process usedGpuMemory, sampled every 30 frames",
        "grid_dwell_s": round(float(dwell.sum()), 1),
        "grid_passage_pairs": len(passage),
    }
