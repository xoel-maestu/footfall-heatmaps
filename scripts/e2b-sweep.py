"""The E2b association sweep on tune, as pre-registered in reports/iterations-phase10.md.

    uv run --no-sync python scripts/e2b-sweep.py [reports/e2b-sweep-tune.json]

Tune scenes 071-075 only (never sealed, never WILDTRACK), every calibrated
camera, per-camera recipe and fusion of configs/e2b-association.yaml from the
cached tracks (CPU only). Each variant is scored against the baseline (every
camera track its own global id) with the paired block bootstrap of mc-IDF1
(30 s blocks, 1000 resamples). The passage side effect is scored on
the whole run only (point values): its EMD makes the bootstrap slow. A1-A3 first; A4 adds the handover to the best
of them by pooled mc-IDF1. The selection rule is the pre-registered one: the
variant with the highest pooled mc-IDF1 among those the acceptance rule
accepts (CI of the delta excludes 0, delta >= .01).
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from space_usage_heatmaps.cli import (
    ASSOCIATION_CONFIG,
    _assoc_dict,
    data_dir,
    load_config,
    load_fusion,
    track_source,
)
from space_usage_heatmaps.evaluation.application.evaluate_identity import evaluate_identity
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.floor_identity import floor_identity
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.occupancy.domain.association import NONE, Association

MIN_DELTA = 0.01
VARIANTS = {
    "A1 colocate, gate 0.5 m": Association("colocate", gate_m=0.5),
    "A2 colocate, gate 1.0 m": Association("colocate", gate_m=1.0),
    "A3 colocate-mnn, gate 1.0 m": Association("colocate-mnn", gate_m=1.0),
}
HANDOVER = {"handover_s": 2.0, "handover_speed_mps": 1.5, "handover_radius_m": 0.5}


def est(e):
    return {"value": e.value, "lo": e.lo, "hi": e.hi}


def score(association):
    reader = NvidiaMtmcReader(data_dir())
    config = load_config(ASSOCIATION_CONFIG)
    r = evaluate_identity(
        reader,
        [s for s, _ in SPLITS["tune"]],
        (config, track_source(config, reader)),
        load_fusion(ASSOCIATION_CONFIG),
        {"associated": association, "baseline": NONE},
        floor_identity,
        oracle=True,
        passage_resamples=1,
    )
    d = r.idf1.delta_pooled
    passage = {
        t: {
            "global": res.pooled[0]["passage"]["mass_error"].value,
            "e2a": res.pooled[1]["passage"]["mass_error"].value,
            "dwell_delta_tvd": res.delta_pooled["dwell"]["tvd"].value,
        }
        for t, res in r.passage.items()
    }
    return {
        "association": _assoc_dict(association),
        "pooled": {n: est(r.idf1.pooled[i]) for i, n in enumerate(r.names)} | {"delta": est(d)},
        "accepted": bool((d.lo > 0 or d.hi < 0) and abs(d.value) >= MIN_DELTA),
        "scenes": {
            s.scene: {
                "idf1": {n: est(r.idf1.scenes[k][i]) for i, n in enumerate(r.names)}
                | {"delta": est(r.idf1.delta_scenes[k])},
                "counts": s.counts,
                "global_ids": s.global_ids,
                "camera_tracks": s.camera_tracks,
                "truth_people": s.truth_people,
            }
            for k, s in enumerate(r.scenes)
        },
        "passage_mass_error": passage,
    }


def main(out):
    results = {}
    with ProcessPoolExecutor(len(VARIANTS)) as pool:
        for name, res in zip(VARIANTS, pool.map(score, VARIANTS.values()), strict=True):
            results[name] = res
            print(f"{name}: {json.dumps(res['pooled'])} accepted {res['accepted']}", flush=True)
    best = max(VARIANTS, key=lambda n: results[n]["pooled"]["associated"]["value"])
    a4 = f"A4 {best.split(' ', 1)[1]} + handover 2 s"
    results[a4] = score(replace(VARIANTS[best], **HANDOVER))
    print(f"{a4}: {json.dumps(results[a4]['pooled'])} accepted {results[a4]['accepted']}")
    accepted = [n for n in results if results[n]["accepted"]]
    chosen = max(accepted, key=lambda n: results[n]["pooled"]["associated"]["value"], default=None)
    report = {
        "split": "tune",
        "scenes": [s for s, _ in SPLITS["tune"]],
        "config": ASSOCIATION_CONFIG,
        "metric": "mc-idf1, floor gate 1.0 m, visible@joint truth, 30 s blocks, 1000 resamples",
        "rule": "highest pooled mc-IDF1 among the variants the acceptance rule accepts",
        "variants": results,
        "best_of_a1_a3": best,
        "chosen": chosen,
    }
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"chosen: {chosen}; {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/e2b-sweep-tune.json")
