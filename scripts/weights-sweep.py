"""Fusion weights (improvements plan, phase 4), as pre-registered in
iterations.md, "Improvements plan, phase 4: fusion weights".

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

Phase 6, H1 (iterations.md, "Improvements plan, phase 6"): the same rule on
partial covered-cell normalisation, alpha in {.25, .5, .75} with p=2 on 2 m
zones (A25, A50, A75), each paired with W1 on the product recipe
configs/e2a-weights.yaml (same tracks as phase 4). Only the two truth blocks the
rule reads are bootstrapped (full@overlap, full@joint; each truth is
bootstrapped on its own, so their values do not change).

    uv run --no-sync python scripts/weights-sweep.py variant A50
    uv run --no-sync python scripts/weights-sweep.py decide h1

SWEEP_WORKERS=n scores the scenes in n processes (default one per scene; 1:
in this process).
"""

import json
import os
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

STRIDE, CACHED_STRIDE = 10, 2
RESAMPLES = 1000
SWEEPS = {
    "phase4": {
        "recipe": "configs/e2a-stitch.yaml",
        "base": "F5",
        "prefix": "weights",
        "truths": ("full@joint", "full@anchor", "full@overlap"),
        "variants": {
            "W1": Fusion("visibility", 2.0, 2.0),
            "W2": Fusion("visibility", 1.0, 2.0, covered_cells=True),
            "W3": Fusion("visibility", 2.0, 2.0, covered_cells=True),
        },
    },
    "h1": {
        "recipe": "configs/e2a-weights.yaml",
        "base": "W1",
        "prefix": "alpha",
        "truths": ("full@joint", "full@overlap"),
        "variants": {
            "A25": Fusion("visibility", 2.0, 2.0, covered_alpha=0.25),
            "A50": Fusion("visibility", 2.0, 2.0, covered_alpha=0.5),
            "A75": Fusion("visibility", 2.0, 2.0, covered_alpha=0.75),
        },
    },
}
VARIANTS = {n: (k, f) for k, sw in SWEEPS.items() for n, f in sw["variants"].items()}
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m")
MARGIN = -0.01
TARGET = 0.10


def est(e):
    return {"value": e.value, "lo": e.lo, "hi": e.hi}


def dwell(t):
    return {m: est(t["dwell"][m]) for m in METRICS}


def base_run(reader, recipe):
    config = replace(load_config(recipe), stride=STRIDE)
    return config, track_source(config, reader, detections_stride=CACHED_STRIDE)


def score(name, split):
    t0 = time.time()
    sweep, fusion = VARIANTS[name]
    recipe, base = SWEEPS[sweep]["recipe"], SWEEPS[sweep]["base"]
    reader = NvidiaMtmcReader(data_dir())
    run, single = base_run(reader, recipe), load_config(BASELINE_CONFIG)
    base_fusion = load_fusion(recipe)
    workers = int(os.environ.get("SWEEP_WORKERS", len(SPLITS[split])))
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        r = evaluate_fusion(
            reader,
            SPLITS[split],
            run,
            (single, track_source(single, reader)),
            0.5,
            fusion,
            RESAMPLES,
            seed=0,
            map=pool.map if pool else map,
            compare_run=run,
            compare_fusion=base_fusion,
            truths=SWEEPS[sweep]["truths"],
        )
    finally:
        if pool:
            pool.shutdown()
    overlap, joint = r.results["full@overlap"], r.results["full@joint"]
    return {
        "split": split,
        "scenes": [s for s, _ in SPLITS[split]],
        "recipe": f"{recipe} at stride {STRIDE} from the stride-{CACHED_STRIDE} detection cache",
        "variant": name,
        "fusion": vars(fusion),
        "baseline": f"{base} " + json.dumps(vars(base_fusion)),
        "n_resamples": RESAMPLES,
        "seed": 0,
        "interval": "paired block bootstrap, 30 s blocks, recentred percentile 95 %",
        "source_sha256": source_sha256(),
        "overlap_mass_error": {
            "pooled": {
                "variant": est(overlap.pooled[0]["dwell"]["mass_error"]),
                base: est(overlap.pooled[1]["dwell"]["mass_error"]),
                "delta": est(overlap.delta_pooled["dwell"]["mass_error"]),
            },
            "scenes": {
                s.scene: {
                    "variant": est(overlap.scenes[i][0]["dwell"]["mass_error"]),
                    base: est(overlap.scenes[i][1]["dwell"]["mass_error"]),
                    "delta": est(overlap.delta_scenes[i]["dwell"]["mass_error"]),
                }
                for i, s in enumerate(r.scenes)
            },
        },
        "tvd_full_joint": {
            "variant": est(joint.pooled[0]["dwell"]["tvd"]),
            base: est(joint.pooled[1]["dwell"]["tvd"]),
            "delta": est(joint.delta_pooled["dwell"]["tvd"]),
        },
        "dwell": {
            truth: {
                "pooled": {
                    "variant": dwell(r.results[truth].pooled[0]),
                    base: dwell(r.results[truth].pooled[1]),
                    "delta": dwell(r.results[truth].delta_pooled),
                }
            }
            for truth in SWEEPS[sweep]["truths"]
        },
        "seconds": round(time.time() - t0),
    }


def line(res, base):
    o, t = res["overlap_mass_error"], res["tvd_full_joint"]
    scenes = ", ".join(f"{s} {v['variant']['value']:.3f}" for s, v in o["scenes"].items())
    d, dt = o["pooled"]["delta"], t["delta"]
    return (
        f"{res['variant']} {res['split']}: overlap mass err {o['pooled']['variant']['value']:.4f} "
        f"({base} {o['pooled'][base]['value']:.4f}) delta {d['value']:+.4f} "
        f"[{d['lo']:+.4f}, {d['hi']:+.4f}]; TVD full@joint {t['variant']['value']:.4f} "
        f"({base} {t[base]['value']:.4f}) delta {dt['value']:+.4f} "
        f"[{dt['lo']:+.4f}, {dt['hi']:+.4f}]; "
        f"scenes {scenes} ({res['seconds']} s)"
    )


def write(out, report):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")


def decide(sweep):
    base, prefix = SWEEPS[sweep]["base"], SWEEPS[sweep]["prefix"]
    reports = {
        n: json.loads(Path(f"reports/{prefix}-{n}-tune.json").read_text())
        for n in SWEEPS[sweep]["variants"]
    }
    eligible = [n for n, r in reports.items() if r["tvd_full_joint"]["delta"]["lo"] <= 0]
    best = min(
        eligible,
        key=lambda n: reports[n]["overlap_mass_error"]["pooled"]["variant"]["value"],
        default=None,
    )
    # F5 in phase 4, W1 in phase 6 H1: the baseline stays when none is selected.
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
        "overlap mass error among eligible, else the baseline; accepted: pooled delta overlap "
        "mass error CI excludes 0 and delta <= -.01",
        "eligible": eligible,
        "baseline": base,
        "selected": best or base,
        "accepted": accepted,
        "target_every_scene_overlap_le_0.10": target,
        "variants": reports,
    }
    write(f"reports/{prefix}-sweep-tune.json", out)
    for r in reports.values():
        print(line(r, base))
    print(f"eligible {eligible}; selected {best or base}; accepted {accepted}; target {target}")


if __name__ == "__main__":
    mode, name = (sys.argv[1:] + [None, None])[:2]
    if mode in ("variant", "sealed") and name in VARIANTS:
        split = "tune" if mode == "variant" else "sealed"
        res = score(name, split)
        sweep = SWEEPS[VARIANTS[name][0]]
        print(line(res, sweep["base"]), flush=True)
        write(f"reports/{sweep['prefix']}-{name}-{split}.json", res)
    elif mode == "decide":
        decide(name or "phase4")
    else:
        raise SystemExit(__doc__)
