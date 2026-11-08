"""Real-time budget, detection stride (improvements plan, phase 3), as
pre-registered in iterations.md, "Improvements plan, phase 3: real-time budget".

    uv run --no-sync python scripts/stride-sweep.py sweep
    uv run --no-sync python scripts/stride-sweep.py sealed <stride>

sweep: on tune 071-075, every calibrated camera, the fused recipe of
configs/e2a-stitch.yaml (E2a fusion plus stitching G10 R1) at effective strides
4, 6, 8, 10 and 16, each paired with the same recipe at stride 2 (the config
unchanged). A stride S replays the cached stride-2 TensorRT detections, keeping
every S/2-th cached frame, with no GPU detection; ByteTrack is built at 30/S fps.
Each stride: fused dwell TVD and passage mass error at full@joint, and the
paired delta (S - 2) with its CI (block bootstrap, 30 s blocks, 1000 resamples,
seed 0). Rule: a stride passes when the upper CI bound of the delta is <= +.01
for both metrics; the chosen stride is the largest passing one, else 2.

sealed: the given stride against stride 2 on 076-080, the same way, once.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from space_usage_heatmaps.cli import (
    BASELINE_CONFIG,
    data_dir,
    load_config,
    load_fusion,
    source_sha256,
    track_source,
)
from space_usage_heatmaps.evaluation.application.evaluate_fusion import evaluate_fusion
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader

RECIPE = "configs/e2a-stitch.yaml"
CACHED_STRIDE = 2
STRIDES = (4, 6, 8, 10, 16)
MARGIN = 0.01
RESAMPLES = 1000
TRUTHS = ("full@joint", "visible@joint", "full@anchor", "full@overlap")
LAYERS = ("dwell", "passage")
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m", "pearson")
WORKERS = 6


def est(e):
    return {"value": e.value, "lo": e.lo, "hi": e.hi}


def table(t):
    return {layer: {m: est(t[layer][m]) for m in METRICS} for layer in LAYERS}


def run_at(stride, reader):
    """(config, track source) of the recipe at an effective stride."""
    config = replace(load_config(RECIPE), stride=stride)
    return config, track_source(config, reader, detections_stride=CACHED_STRIDE)


def warm(item):
    """The tracks of one camera at one stride, into their cache."""
    stride, scene, camera = item
    t0 = time.time()
    _, source = run_at(stride, NvidiaMtmcReader(data_dir()))
    n = len(source.tracks(scene, camera).track_ids)
    return stride, scene, camera, n, round(time.time() - t0)


def warm_all(strides, split):
    reader = NvidiaMtmcReader(data_dir())
    items = [
        (s, scene, str(c))
        for s in strides
        for scene, _ in SPLITS[split]
        for c in reader.cameras(scene)
    ]
    with ProcessPoolExecutor(WORKERS) as pool:
        for stride, scene, camera, n, sec in pool.map(warm, items):
            print(
                f"tracks stride {stride} scene {scene} camera {camera}: {n} rows ({sec} s)",
                flush=True,
            )


def score(item):
    """Stride S against stride 2 on one split: the paired fused evaluation."""
    stride, split = item
    t0 = time.time()
    reader = NvidiaMtmcReader(data_dir())
    single = load_config(BASELINE_CONFIG)
    r = evaluate_fusion(
        reader,
        SPLITS[split],
        run_at(stride, reader),
        (single, track_source(single, reader)),
        0.5,
        load_fusion(RECIPE),
        RESAMPLES,
        seed=0,
        compare_run=run_at(CACHED_STRIDE, reader),
    )
    joint = r.results["full@joint"]
    d_tvd = joint.delta_pooled["dwell"]["tvd"]
    d_mass = joint.delta_pooled["passage"]["mass_error"]
    out = {
        "stride": stride,
        "fps_per_camera": 30 / stride,
        "dwell_tvd_full_joint": est(joint.pooled[0]["dwell"]["tvd"]),
        "passage_mass_error_full_joint": est(joint.pooled[0]["passage"]["mass_error"]),
        "stride2_dwell_tvd_full_joint": est(joint.pooled[1]["dwell"]["tvd"]),
        "stride2_passage_mass_error_full_joint": est(joint.pooled[1]["passage"]["mass_error"]),
        "delta_dwell_tvd_full_joint": est(d_tvd),
        "delta_passage_mass_error_full_joint": est(d_mass),
        "passes": bool(d_tvd.hi <= MARGIN and d_mass.hi <= MARGIN),
        "camera_tracks": {s.scene: s.camera_tracks for s in r.scenes},
        "results": {
            truth: {
                "pooled": {
                    "stride": table(res.pooled[0]),
                    "stride2": table(res.pooled[1]),
                    "delta": table(res.delta_pooled),
                },
                "scenes": {
                    x.scene: {
                        "stride": table(res.scenes[i][0]),
                        "stride2": table(res.scenes[i][1]),
                        "delta": table(res.delta_scenes[i]),
                    }
                    for i, x in enumerate(r.scenes)
                },
            }
            for truth, res in r.results.items()
            if truth in TRUTHS
        },
        "seconds": round(time.time() - t0),
    }
    return out


def line(res):
    t, m = res["delta_dwell_tvd_full_joint"], res["delta_passage_mass_error_full_joint"]
    return (
        f"stride {res['stride']}: dwell TVD {res['dwell_tvd_full_joint']['value']:.4f} "
        f"(stride 2 {res['stride2_dwell_tvd_full_joint']['value']:.4f}) delta {t['value']:+.4f} "
        f"[{t['lo']:+.4f}, {t['hi']:+.4f}]; passage mass err "
        f"{res['passage_mass_error_full_joint']['value']:.4f} "
        f"(stride 2 {res['stride2_passage_mass_error_full_joint']['value']:.4f}) delta "
        f"{m['value']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]; passes {res['passes']} "
        f"({res['seconds']} s)"
    )


def header(split):
    return {
        "split": split,
        "scenes": [s for s, _ in SPLITS[split]],
        "recipe": RECIPE,
        "cached_detections": f"yolox-s-960-onnx-trt-stride{CACHED_STRIDE} (stride 1 where cached)",
        "tracker_frame_rate": "video fps / stride",
        "n_resamples": RESAMPLES,
        "seed": 0,
        "interval": "paired block bootstrap, 30 s blocks, recentred percentile 95 %",
        "rule": "a stride passes when the upper CI bound of delta (S - 2) is <= +.01 for "
        "fused dwell TVD and passage mass error at full@joint; chosen: the largest passing, else 2",
        "source_sha256": source_sha256(),
    }


def write(out, report):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")


def sweep(out):
    t0 = time.time()
    warm_all(STRIDES, "tune")
    print(f"tracks cached ({round(time.time() - t0)} s)", flush=True)
    results = {}
    with ProcessPoolExecutor(len(STRIDES)) as pool:
        for res in pool.map(score, [(s, "tune") for s in STRIDES]):
            results[res["stride"]] = res
            print(line(res), flush=True)
    passing = [s for s in STRIDES if results[s]["passes"]]
    chosen = max(passing, default=CACHED_STRIDE)
    write(
        out,
        header("tune")
        | {
            "strides": {str(s): r for s, r in results.items()},
            "chosen": chosen,
            "seconds": round(time.time() - t0),
        },
    )
    print(f"chosen stride: {chosen}; {out}")


def sealed(stride, out):
    t0 = time.time()
    warm_all((stride,), "sealed")
    res = score((stride, "sealed"))
    print(line(res), flush=True)
    write(out, header("sealed") | {"stride": res, "seconds": round(time.time() - t0)})
    print(out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sweep("reports/stride-sweep-tune.json")
    elif mode == "sealed" and len(sys.argv) > 2 and int(sys.argv[2]) in STRIDES:
        sealed(int(sys.argv[2]), f"reports/stride-sealed-{sys.argv[2]}.json")
    else:
        raise SystemExit(__doc__)
