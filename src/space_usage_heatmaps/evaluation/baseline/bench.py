"""Sustained end-to-end speed of the baseline (evaluation.infrastructure.bench)."""

import torch

from space_usage_heatmaps.evaluation.baseline.ultralytics_tracks import (
    load_model,
    track_frames,
    video_frames,
)
from space_usage_heatmaps.evaluation.infrastructure import bench as sustained


def bench(config, video, cam, spec, minutes, weights_dir):
    model = load_model(config, weights_dir)
    tracked = track_frames(model, config, video_frames(video, config.stride, True))
    result = sustained.bench(tracked, config, cam, spec, minutes)
    engine = f"{config.engine} (PyTorch {torch.__version__}, CUDA, fp32)"
    return {"engine": engine} | result
