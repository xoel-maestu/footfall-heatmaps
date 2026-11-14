"""Intra-camera tracklet stitching (improvements plan, phase 2), as pre-registered
in docs/experiment-log.md, "Improvements plan, phase 2: tracklet stitching".

    uv run --no-sync python scripts/stitch-sweep.py sweep
    uv run --no-sync python scripts/stitch-sweep.py confirm tune|sealed

sweep: the six variants G in {2, 5, 10} s x R in {0.5, 1.0} m (v_max 1.5 m/s)
on tune 071-075 only, every calibrated camera, per-camera recipe and fusion of
configs/e2a-fusion.yaml from the cached tracks (CPU only). Point values of the
whole run (draw 0 of the block bootstrap), pooled as the unweighted mean over
scenes, each paired with the baseline (the same fused map without stitching).
Guards per variant: fused dwell byte-identical to the baseline, and HOTA/IDF1
of each scene's E1 camera (configs/h5-best.yaml with the same stitching
against without) at most 1 point lower. Selection: the lowest pooled fused
passage mass error full@joint among the variants that pass the guards.

confirm: configs/e2a-stitch.yaml (the selected variant) against
configs/e2a-fusion.yaml on one split, paired block bootstrap, 1000 resamples,
seed 0, and the same guards with configs/h5-stitch.yaml against
configs/h5-best.yaml. Sealed is scored once, unchanged.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from footfall_heatmaps.cli import (
    BASELINE_CONFIG,
    FUSION_CONFIG,
    data_dir,
    load_config,
    load_fusion,
    score_tracking,
    source_sha256,
    track_source,
)
from footfall_heatmaps.evaluation.application.evaluate_fusion import evaluate_fusion
from footfall_heatmaps.evaluation.domain.splits import SPLITS
from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader

STITCH_CONFIG = "configs/e2a-stitch.yaml"
E1_STITCH_CONFIG = "configs/h5-stitch.yaml"
V_MAX = 1.5
VARIANTS = {
    f"G{g:g} R{r:g}": {"gap_s": float(g), "radius_m": r, "v_max_mps": V_MAX}
    for g in (2, 5, 10)
    for r in (0.5, 1.0)
}
TARGET = 0.30
MIN_DELTA = 0.01
GUARD_POINTS = 1.0
TRUTHS = ("full@joint", "visible@joint", "full@anchor", "full@overlap")
LAYERS = ("dwell", "passage")
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m", "pearson")


def est(e):
    return {"value": e.value, "lo": e.lo, "hi": e.hi}


def table(t):
    return {layer: {m: est(t[layer][m]) for m in METRICS} for layer in LAYERS}


def dwell_delta_is_zero(r):
    """Every dwell metric's paired delta (value and interval) is exactly 0."""
    return all(
        all(v in (0.0, None) for v in (e.value, e.lo, e.hi))
        for res in r.results.values()
        for e in res.delta_pooled["dwell"].values()
    ) and all(
        all(v in (0.0, None) for v in (e.value, e.lo, e.hi))
        for res in r.results.values()
        for d in res.delta_scenes
        for e in d["dwell"].values()
    )


def fused(stitched, base, single, scenes, resamples, pool_map=map):
    """evaluate_fusion of the stitched recipe paired with the plain one."""
    reader = NvidiaMtmcReader(data_dir())
    runs = [(c, track_source(c, reader)) for c in (stitched, single, base)]
    return evaluate_fusion(
        reader,
        scenes,
        runs[0],
        runs[1],
        0.5,
        load_fusion(FUSION_CONFIG),
        resamples,
        map=pool_map,
        compare_run=runs[2],
    )


def tracking(config, scenes):
    """HOTA, DetA, AssA, IDF1 of one E1 recipe per scene and pooled."""
    reader = NvidiaMtmcReader(data_dir())
    t = score_tracking(reader, scenes, ["x"], [(config, track_source(config, reader))])
    return {"scenes": t["scenes"]["x"], "pooled": t["pooled"]["x"]}


def tracking_guard(stitched, plain):
    """HOTA/IDF1 of the E1 camera with stitching against without: at most 1 point lower."""
    d = {m: stitched["pooled"][m] - plain["pooled"][m] for m in plain["pooled"]}
    return {
        "stitched": stitched,
        "plain": plain,
        "delta_pooled": d,
        "guard_holds": bool(100 * min(d["hota"], d["idf1"]) >= -GUARD_POINTS),
    }


def summary(r, base_tracks):
    joint = r.results["full@joint"]
    return {
        "passage_mass_error_full_joint": joint.pooled[0]["passage"]["mass_error"].value,
        "baseline_passage_mass_error_full_joint": joint.pooled[1]["passage"]["mass_error"].value,
        "delta_passage_mass_error_full_joint": est(joint.delta_pooled["passage"]["mass_error"]),
        "dwell_tvd_full_joint": joint.pooled[0]["dwell"]["tvd"].value,
        "baseline_dwell_tvd_full_joint": joint.pooled[1]["dwell"]["tvd"].value,
        "camera_tracks": {s.scene: s.camera_tracks for s in r.scenes},
        "baseline_camera_tracks": base_tracks,
        "scenes": {
            s.scene: {
                "passage_mass_error_full_joint": joint.scenes[i][0]["passage"]["mass_error"].value,
                "baseline": joint.scenes[i][1]["passage"]["mass_error"].value,
            }
            for i, s in enumerate(r.scenes)
        },
    }


def score_variant(item):
    name, stitch, plain = item
    t0 = time.time()
    base, single = load_config(FUSION_CONFIG), load_config(BASELINE_CONFIG)
    scenes = SPLITS["tune"]
    r = fused(replace(base, stitch=stitch), base, single, scenes, 1)
    same_dwell = all(
        r.fused[s].dwell.dtype == plain["dwell"][s].dtype
        and np.array_equal(r.fused[s].dwell, plain["dwell"][s])
        for s, _ in scenes
    )
    track = tracking_guard(tracking(replace(single, stitch=stitch), scenes), plain["tracking"])
    out = summary(r, plain["camera_tracks"]) | {
        "stitch": stitch,
        "guards": {
            "fused_dwell_byte_identical": bool(same_dwell),
            "dwell_delta_zero": dwell_delta_is_zero(r),
            "e1_tracking": track,
        },
        "seconds": round(time.time() - t0),
    }
    g = out["guards"]
    out["passes_guards"] = bool(
        g["fused_dwell_byte_identical"] and g["dwell_delta_zero"] and track["guard_holds"]
    )
    return name, out


def plain_run(split):
    """The baseline's fused dwell grids, track counts and E1 tracking, once."""
    base, single = load_config(FUSION_CONFIG), load_config(BASELINE_CONFIG)
    scenes = SPLITS[split]
    r = fused(base, base, single, scenes, 1)
    return {
        "dwell": {s: g.dwell for s, g in r.fused.items()},
        "camera_tracks": {s.scene: s.camera_tracks for s in r.scenes},
        "tracking": tracking(single, scenes),
    }


def sweep(out):
    t0 = time.time()
    plain = plain_run("tune")
    print(f"baseline scored ({round(time.time() - t0)} s)", flush=True)
    results = {}
    items = [(n, s, plain) for n, s in VARIANTS.items()]
    with ProcessPoolExecutor(3) as pool:
        for name, res in pool.map(score_variant, items):
            results[name] = res
            d = res["guards"]["e1_tracking"]["delta_pooled"]
            print(
                f"{name}: passage mass err full@joint {res['passage_mass_error_full_joint']:.4f} "
                f"(baseline {res['baseline_passage_mass_error_full_joint']:.4f}), dwell TVD "
                f"{res['dwell_tvd_full_joint']:.4f}, ids {sum(res['camera_tracks'].values())} "
                f"(baseline {sum(res['baseline_camera_tracks'].values())}), E1 dHOTA "
                f"{100 * d['hota']:+.2f} dIDF1 {100 * d['idf1']:+.2f}, guards "
                f"{res['passes_guards']} ({res['seconds']} s)",
                flush=True,
            )
    passing = [n for n in results if results[n]["passes_guards"]]
    chosen = min(passing, key=lambda n: results[n]["passage_mass_error_full_joint"], default=None)
    report = {
        "split": "tune",
        "scenes": [s for s, _ in SPLITS["tune"]],
        "recipe": FUSION_CONFIG,
        "e1_recipe": BASELINE_CONFIG,
        "values": "whole-run point values, no CI; pooled = unweighted mean over scenes",
        "rule": "lowest pooled fused passage mass error full@joint among the variants "
        "that pass the guards (fused dwell byte-identical, E1 HOTA/IDF1 drop <= 1 point)",
        "source_sha256": source_sha256(),
        "variants": results,
        "chosen": chosen,
        "seconds": round(time.time() - t0),
    }
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"chosen: {chosen}; {out}")


def confirm(split, out):
    t0 = time.time()
    stitched, base = load_config(STITCH_CONFIG), load_config(FUSION_CONFIG)
    single = load_config(BASELINE_CONFIG)
    if stitched.stitch is None or replace(stitched, stitch=None) != base:
        raise SystemExit(f"{STITCH_CONFIG} must be {FUSION_CONFIG}'s recipe plus a stitch section")
    scenes = SPLITS[split]
    e1 = load_config(E1_STITCH_CONFIG)
    if replace(e1, stitch=None) != single or e1.stitch != stitched.stitch:
        raise SystemExit(f"{E1_STITCH_CONFIG} must be {BASELINE_CONFIG} plus the same stitch")
    plain = plain_run(split)
    with ProcessPoolExecutor(len(scenes)) as pool:
        r = fused(stitched, base, single, scenes, 1000, pool.map)
    same_dwell = all(np.array_equal(r.fused[s].dwell, plain["dwell"][s]) for s, _ in scenes)
    track = tracking_guard(tracking(e1, scenes), plain["tracking"])
    joint = r.results["full@joint"].delta_pooled["passage"]["mass_error"]
    accepted = bool((joint.lo > 0 or joint.hi < 0) and abs(joint.value) >= MIN_DELTA)
    s = summary(r, plain["camera_tracks"])
    report = {
        "split": split,
        "scenes": [x for x, _ in scenes],
        "configs": {"stitched": STITCH_CONFIG, "plain": FUSION_CONFIG},
        "stitch": stitched.stitch,
        "n_resamples": 1000,
        "seed": 0,
        "interval": "paired block bootstrap, 30 s blocks, recentred percentile 95 %",
        "source_sha256": source_sha256(),
        **s,
        "target": {
            "passage_mass_error_full_joint_max": TARGET,
            "met": s["passage_mass_error_full_joint"] <= TARGET,
        },
        "accepted": accepted,
        "guards": {
            "fused_dwell_byte_identical": bool(same_dwell),
            "dwell_delta_zero": dwell_delta_is_zero(r),
            "e1_tracking": track,
        },
        "results": {
            truth: {
                "pooled": {
                    "stitched": table(res.pooled[0]),
                    "plain": table(res.pooled[1]),
                    "delta": table(res.delta_pooled),
                },
                "scenes": {
                    x.scene: {
                        "stitched": table(res.scenes[i][0]),
                        "plain": table(res.scenes[i][1]),
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
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    d = track["delta_pooled"]
    print(
        f"{split}: passage mass err full@joint {s['passage_mass_error_full_joint']:.3f} "
        f"vs {s['baseline_passage_mass_error_full_joint']:.3f}, delta {joint.value:+.3f} "
        f"[{joint.lo:+.3f}, {joint.hi:+.3f}], accepted {accepted}, target <= {TARGET} "
        f"{'met' if report['target']['met'] else 'missed'}; dwell identical {same_dwell}, "
        f"E1 dHOTA {100 * d['hota']:+.2f} dIDF1 {100 * d['idf1']:+.2f}; {out}"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sweep(sys.argv[2] if len(sys.argv) > 2 else "reports/stitch-sweep-tune.json")
    elif mode == "confirm" and len(sys.argv) > 2 and sys.argv[2] in ("tune", "sealed"):
        confirm(sys.argv[2], f"reports/stitch-confirm-{sys.argv[2]}.json")
    else:
        raise SystemExit(__doc__)
