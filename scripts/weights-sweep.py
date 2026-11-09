"""Fusion weights (improvements plan, phase 4), as pre-registered in
the experiment log.

    uv run --no-sync python scripts/weights-sweep.py variant W2     # one variant on tune
    uv run --no-sync python scripts/weights-sweep.py decide         # the rule, from the 3 reports
    uv run --no-sync python scripts/weights-sweep.py sealed W2      # the chosen one, once

Base: the phase 3 product recipe, configs/e2a-stitch.yaml at stride 10 from the
cached stride-2 TensorRT detections (no GPU), every calibrated camera. Each
variant is fused from the same tracks as F5 (configs/e2a-stitch.yaml's fusion:
visibility p=1 on 2 m zones) and paired with it (block bootstrap, 30 s blocks,
1000 resamples, seed 0): dwell mass error at full@overlap (pooled and per
scene) and dwell TVD at full@joint.

Selection: among the variants whose delta TVD full@joint CI has a lower bound
<= 0, the lowest pooled overlap mass error; none: F5 stays. Acceptance: the
pooled delta overlap mass error has a CI that excludes 0, with delta <= -.01.
Target (reported, not a gate): overlap mass error <= .10 on every tune scene.
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
from space_usage_heatmaps.occupancy.domain.fusion import Fusion

RECIPE = "configs/e2a-stitch.yaml"
STRIDE, CACHED_STRIDE = 10, 2
RESAMPLES = 1000
VARIANTS = {
    "W1": Fusion("visibility", 2.0, 2.0),
    "W2": Fusion("visibility", 1.0, 2.0, covered_cells=True),
    "W3": Fusion("visibility", 2.0, 2.0, covered_cells=True),
}
TRUTHS = ("full@joint", "full@anchor", "full@overlap")
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m")
MARGIN = -0.01
TARGET = 0.10


def est(e):
    return {"value": e.value, "lo": e.lo, "hi": e.hi}


def dwell(t):
    return {m: est(t["dwell"][m]) for m in METRICS}


def base_run(reader):
    config = replace(load_config(RECIPE), stride=STRIDE)
    return config, track_source(config, reader, detections_stride=CACHED_STRIDE)


def score(name, split):
    t0 = time.time()
    reader = NvidiaMtmcReader(data_dir())
    run, single = base_run(reader), load_config(BASELINE_CONFIG)
    f5 = load_fusion(RECIPE)
    with ProcessPoolExecutor(len(SPLITS[split])) as pool:
        r = evaluate_fusion(
            reader,
            SPLITS[split],
            run,
            (single, track_source(single, reader)),
            0.5,
            VARIANTS[name],
            RESAMPLES,
            seed=0,
            map=pool.map,
            compare_run=run,
            compare_fusion=f5,
        )
    overlap, joint = r.results["full@overlap"], r.results["full@joint"]
    return {
        "split": split,
        "scenes": [s for s, _ in SPLITS[split]],
        "recipe": f"{RECIPE} at stride {STRIDE} from the stride-{CACHED_STRIDE} detection cache",
        "variant": name,
        "fusion": vars(VARIANTS[name]),
        "baseline": "F5 " + json.dumps(vars(f5)),
        "n_resamples": RESAMPLES,
        "seed": 0,
        "interval": "paired block bootstrap, 30 s blocks, recentred percentile 95 %",
        "source_sha256": source_sha256(),
        "overlap_mass_error": {
            "pooled": {
                "variant": est(overlap.pooled[0]["dwell"]["mass_error"]),
                "F5": est(overlap.pooled[1]["dwell"]["mass_error"]),
                "delta": est(overlap.delta_pooled["dwell"]["mass_error"]),
            },
            "scenes": {
                s.scene: {
                    "variant": est(overlap.scenes[i][0]["dwell"]["mass_error"]),
                    "F5": est(overlap.scenes[i][1]["dwell"]["mass_error"]),
                    "delta": est(overlap.delta_scenes[i]["dwell"]["mass_error"]),
                }
                for i, s in enumerate(r.scenes)
            },
        },
        "tvd_full_joint": {
            "variant": est(joint.pooled[0]["dwell"]["tvd"]),
            "F5": est(joint.pooled[1]["dwell"]["tvd"]),
            "delta": est(joint.delta_pooled["dwell"]["tvd"]),
        },
        "dwell": {
            truth: {
                "pooled": {
                    "variant": dwell(r.results[truth].pooled[0]),
                    "F5": dwell(r.results[truth].pooled[1]),
                    "delta": dwell(r.results[truth].delta_pooled),
                }
            }
            for truth in TRUTHS
        },
        "seconds": round(time.time() - t0),
    }


def line(res):
    o, t = res["overlap_mass_error"], res["tvd_full_joint"]
    scenes = ", ".join(f"{s} {v['variant']['value']:.3f}" for s, v in o["scenes"].items())
    d, dt = o["pooled"]["delta"], t["delta"]
    return (
        f"{res['variant']} {res['split']}: overlap mass err {o['pooled']['variant']['value']:.4f} "
        f"(F5 {o['pooled']['F5']['value']:.4f}) delta {d['value']:+.4f} "
        f"[{d['lo']:+.4f}, {d['hi']:+.4f}]; TVD full@joint {t['variant']['value']:.4f} "
        f"(F5 {t['F5']['value']:.4f}) delta {dt['value']:+.4f} [{dt['lo']:+.4f}, {dt['hi']:+.4f}]; "
        f"scenes {scenes} ({res['seconds']} s)"
    )


def write(out, report):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")


def decide():
    reports = {n: json.loads(Path(f"reports/weights-{n}-tune.json").read_text()) for n in VARIANTS}
    eligible = [n for n, r in reports.items() if r["tvd_full_joint"]["delta"]["lo"] <= 0]
    best = min(
        eligible,
        key=lambda n: reports[n]["overlap_mass_error"]["pooled"]["variant"]["value"],
        default=None,
    )
    accepted = False
    if best is not None:
        d = reports[best]["overlap_mass_error"]["pooled"]["delta"]
        accepted = bool((d["hi"] < 0 or d["lo"] > 0) and d["value"] <= MARGIN)
    target = {
        n: all(v["variant"]["value"] <= TARGET for v in r["overlap_mass_error"]["scenes"].values())
        for n, r in reports.items()
    }
    out = {
        "rule": "eligible: delta TVD full@joint CI lower bound <= 0; selected: lowest pooled "
        "overlap mass error among eligible, else F5; accepted: pooled delta overlap mass error "
        "CI excludes 0 and delta <= -.01",
        "eligible": eligible,
        "selected": best or "F5",
        "accepted": accepted,
        "target_every_scene_overlap_le_0.10": target,
        "variants": reports,
    }
    write("reports/weights-sweep-tune.json", out)
    for r in reports.values():
        print(line(r))
    print(f"eligible {eligible}; selected {best or 'F5'}; accepted {accepted}; target {target}")


if __name__ == "__main__":
    mode, name = (sys.argv[1:] + [None, None])[:2]
    if mode in ("variant", "sealed") and name in VARIANTS:
        split = "tune" if mode == "variant" else "sealed"
        res = score(name, split)
        print(line(res), flush=True)
        write(f"reports/weights-{name}-{split}.json", res)
    elif mode == "decide":
        decide()
    else:
        raise SystemExit(__doc__)
