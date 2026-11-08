"""Re-score the E2a fusion sweep (F1-F5 and the explored zone variants) on tune.

    uv run --no-sync python scripts/e2a-sweep.py [reports/e2a-sweep-tune.json]

Tune scenes 071-075 only (never sealed), every calibrated camera, per-camera
recipe configs/e2a-stride2.yaml from the cached tracks (CPU only; a missing
track cache would run the detector). Point values of the whole run (draw 0 of
the block bootstrap; no CI), pooled as the unweighted mean over scenes, as in
the sweep rows of reports/iterations-phase9.md.
"""

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from space_usage_heatmaps.cli import BASELINE_CONFIG, data_dir, load_config, track_source
from space_usage_heatmaps.evaluation.application.evaluate_fusion import evaluate_fusion
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.occupancy.domain.fusion import Fusion

CONFIG = "configs/e2a-stride2.yaml"
VARIANTS = {
    "F1 mean": Fusion("mean"),
    "F2 visibility p=1": Fusion("visibility", 1.0),
    "F3 visibility p=2": Fusion("visibility", 2.0),
    "F4 visibility p=inf": Fusion("visibility", math.inf),
    "F5 visibility p=1, 2 m zones": Fusion("visibility", 1.0, 2.0),
    "explored: visibility p=2, 2 m zones": Fusion("visibility", 2.0, 2.0),
    "explored: visibility p=inf, 2 m zones": Fusion("visibility", math.inf, 2.0),
}
TRUTHS = ("full@joint", "full@anchor", "full@overlap")
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m")


def point(table, metric):
    v = table["dwell"][metric].value
    return None if v is None else round(float(v), 4)


def main(out):
    reader = NvidiaMtmcReader(data_dir())
    scenes = SPLITS["tune"]
    runs = [(c, track_source(c, reader)) for c in map(load_config, (CONFIG, BASELINE_CONFIG))]
    variants = {}
    with ProcessPoolExecutor(len(scenes)) as pool:
        for name, fusion in VARIANTS.items():
            r = evaluate_fusion(reader, scenes, *runs, 0.5, fusion, 1, map=pool.map)
            variants[name] = {
                "fusion": {
                    "weighting": fusion.weighting,
                    "power": "inf" if math.isinf(fusion.power) else fusion.power,
                    "zone_m": fusion.zone_m,
                },
                "pooled": {
                    t: {m: point(r.results[t].pooled[0], m) for m in METRICS} for t in TRUTHS
                },
                "scenes": {
                    s.scene: {
                        t: {m: point(r.results[t].scenes[i][0], m) for m in METRICS} for t in TRUTHS
                    }
                    for i, s in enumerate(r.scenes)
                },
            }
            joint = variants[name]["pooled"]["full@joint"]
            overlap = variants[name]["pooled"]["full@overlap"]["mass_error"]
            print(
                f"{name}: full@joint mass {joint['mass_error']} tvd {joint['tvd']} "
                f"emd {joint['emd_m']}; full@anchor tvd "
                f"{variants[name]['pooled']['full@anchor']['tvd']}; overlap mass {overlap}",
                flush=True,
            )
    report = {
        "split": "tune",
        "scenes": [s for s, _ in scenes],
        "config": CONFIG,
        "cell_m": 0.5,
        "layer": "dwell",
        "values": "whole-run point values, no CI; pooled = unweighted mean over scenes",
        "variants": variants,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/e2a-sweep-tune.json")
