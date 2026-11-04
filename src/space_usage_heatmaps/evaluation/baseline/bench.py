"""Sustained end-to-end speed of the baseline: decode -> detect -> track -> floor grid.

fps counts source-video frames per wall-clock second (with stride N, each
processed frame stands for N), from the first processed frame on (model load
and the first inference are excluded). fps_p5 is the 5th percentile of the
per-second rates. VRAM is this process's GPU memory as the driver reports it
(NVML), the peak of one sample every 30 frames: it includes the CUDA context,
cuDNN workspaces and PyTorch's cached blocks, which torch's own
max_memory_allocated leaves out, and is what another process on the 4 GB card
would find taken.
"""

import os
import time

import numpy as np
import pynvml
import torch

from space_usage_heatmaps.evaluation.baseline.ultralytics_tracks import (
    load_model,
    track_frames,
    video_frames,
)
from space_usage_heatmaps.floorplan.domain.foot_point import image_points
from space_usage_heatmaps.floorplan.domain.homography import Homography

VRAM_EVERY = 30  # frames between two NVML samples


def process_vram_mb(handle):
    used = [
        p.usedGpuMemory
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if p.pid == os.getpid() and p.usedGpuMemory is not None
    ]
    return sum(used) / 2**20


def bench(config, video, world_to_image, spec, fps, minutes, weights_dir):
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return _run(config, video, world_to_image, spec, fps, minutes, weights_dir, handle)
    finally:
        pynvml.nvmlShutdown()


def _run(config, video, world_to_image, spec, fps, minutes, weights_dir, handle):
    model = load_model(config, weights_dir)
    homography = Homography(world_to_image)
    dwell = np.zeros(spec.ny * spec.nx)
    passage = set()  # (track id, cell)
    rates, peak, frames = [], 0.0, 0
    start = window = None
    in_window = 0
    for _, ids, boxes in track_frames(model, config, video_frames(video, config.stride, True)):
        xs, ys = homography.to_world(*image_points(boxes, config.foot_point))
        col = np.floor((xs - spec.origin_x) / spec.cell_m).astype(int)
        row = np.floor((ys - spec.origin_y) / spec.cell_m).astype(int)
        inside = (col >= 0) & (col < spec.nx) & (row >= 0) & (row < spec.ny)
        cell = row[inside] * spec.nx + col[inside]
        np.add.at(dwell, cell, config.stride / fps)
        passage.update(zip(ids[inside].tolist(), cell.tolist(), strict=True))

        now = time.monotonic()
        if start is None:  # timing starts after the first frame
            start = window = now
            peak = process_vram_mb(handle)
            continue
        frames += 1
        in_window += 1
        if now - window >= 1.0:
            rates.append(in_window * config.stride / (now - window))
            window, in_window = now, 0
        if frames % VRAM_EVERY == 0:
            peak = max(peak, process_vram_mb(handle))
        if now - start >= minutes * 60:
            break
    elapsed = time.monotonic() - start
    peak = max(peak, process_vram_mb(handle))
    return {
        "fps_mean": round(frames * config.stride / elapsed, 2),
        "fps_p5": round(float(np.percentile(rates, 5)), 2) if rates else None,
        "vram_peak_mb": round(peak, 1),
        "engine": f"{config.engine} (PyTorch {torch.__version__}, CUDA, fp32)",
        "frames": frames * config.stride,
        "seconds": round(elapsed, 1),
        "vram_source": "NVML process usedGpuMemory, sampled every 30 frames",
        "grid_dwell_s": round(float(dwell.sum()), 1),
        "grid_passage_pairs": len(passage),
    }
