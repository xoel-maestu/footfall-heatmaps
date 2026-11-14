"""Sustained speed of N cameras sharing one GPU, served in turn (round robin).

Each camera is an iterator that does the work of one processed frame per
next() (decode the frame, detect, track, fill the skipped frames, floor
points, grid) and yields. One round steps every camera once. Timing starts
after the first round (model load and the first inferences are excluded).

Rates count processed frames, not source frames: with stride S, a camera keeps
up with a 30 fps source when it processes 30 / S frames a second. Per camera,
processed_fps_mean is its frames over the timed seconds, and its p5 is the 5th
percentile of its per-second rates. The pooled processed_fps_p5 is the lowest
camera's p5. vram_mb() is sampled once per round; the peak is reported.
"""

import time

import numpy as np


def round_robin(steps, seconds, clock=time.monotonic, vram_mb=lambda: 0.0):
    """steps: one iterator per camera; seconds: timed duration after warm-up."""
    n = len(steps)
    for s in steps:  # warm-up round
        next(s)
    peak = vram_mb()
    start = window = clock()
    total = np.zeros(n, dtype=int)
    in_window = np.zeros(n, dtype=int)
    rates = []
    while True:
        for c, s in enumerate(steps):
            next(s)
            total[c] += 1
            in_window[c] += 1
        peak = max(peak, vram_mb())
        now = clock()
        if now - window >= 1.0:
            rates.append(in_window / (now - window))
            window, in_window = now, np.zeros(n, dtype=int)
        if now - start >= seconds:
            break
    elapsed = clock() - start
    rates = np.array(rates).reshape(-1, n)
    per_camera = [
        {
            "processed_fps_mean": round(float(total[c] / elapsed), 2),
            "processed_fps_p5": round(float(np.percentile(rates[:, c], 5)), 2),
            "processed_frames": int(total[c]),
        }
        for c in range(n)
    ]
    return {
        "cameras": n,
        "processed_fps_mean": round(float(np.mean(total / elapsed)), 2),
        "processed_fps_p5": min(p["processed_fps_p5"] for p in per_camera),
        "vram_peak_mb": round(float(peak), 1),
        "seconds": round(float(elapsed), 1),
        "rounds": int(total.min()),
        "per_camera": per_camera,
    }
