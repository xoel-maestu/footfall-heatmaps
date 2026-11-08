"""Identity-free passage by chord length (improvements plan phase 6, H2), as
pre-registered in iterations.md, "Improvements plan, phase 6".

    uv run --no-sync python scripts/chord-sweep.py baseline   # ids s10 vs ids s2, first
    uv run --no-sync python scripts/chord-sweep.py chord10    # chord s10 vs ids s10
    uv run --no-sync python scripts/chord-sweep.py chord2     # chord s10 vs chord s2
    uv run --no-sync python scripts/chord-sweep.py dwell      # dwell bytes, ids vs chord
    uv run --no-sync python scripts/chord-sweep.py decide     # the rule, from the reports

Base: the product recipe configs/e2a-weights.yaml (W1) with stitching off, on
tune 071-075, every calibrated camera, from the stride-2 TensorRT detection
cache (stride 10 replays it, no GPU). Paired block bootstrap, 30 s blocks,
1000 resamples, seed 0, full@joint only (the truth block the rule reads).

Rule: accept when Delta passage mass error full@joint (chord s10 - ids s10)
has a CI that excludes 0 with Delta <= -.01, the stride invariance
|passage mass error(chord s10) - passage mass error(chord s2)| <= .05 holds
(point values), passage TVD full@joint does not get worse (Delta upper bound
<= +.01) and dwell is byte-identical. Target (reported): chord s10 passage
mass error <= .30.
"""

import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from space_usage_heatmaps.cli import (
    BASELINE_CONFIG,
    data_dir,
    load_config,
    load_fusion,
    source_sha256,
    track_source,
)
from space_usage_heatmaps.evaluation.application.evaluate_config import BLOCK_S
from space_usage_heatmaps.evaluation.application.evaluate_fusion import evaluate_fusion
from space_usage_heatmaps.evaluation.application.evaluate_oracle import recipe_floor_points
from space_usage_heatmaps.evaluation.domain.bootstrap import block_layers
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.grid import GridSpec

RECIPE = "configs/e2a-weights.yaml"
CACHED_STRIDE = 2
RESAMPLES = 1000
TRUTH = "full@joint"
LAYERS = ("dwell", "passage")
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m", "pearson")
MARGIN, INVARIANCE, GUARD, TARGET = -0.01, 0.05, 0.01, 0.30
# name: (first prediction, second prediction), each (passage, stride)
PAIRS = {
    "baseline": (("ids", 10), ("ids", 2)),
    "chord10": (("chord", 10), ("ids", 10)),
    "chord2": (("chord", 10), ("chord", 2)),
}


def est(e):
    return {"value": e.value, "lo": e.lo, "hi": e.hi}


def table(t):
    return {layer: {m: est(t[layer][m]) for m in METRICS} for layer in LAYERS}


def run(passage, stride, reader):
    """(config, track source): the recipe without stitching, at a stride."""
    config = replace(load_config(RECIPE), stride=stride, stitch=None, passage=passage)
    cached = None if stride == CACHED_STRIDE else CACHED_STRIDE
    return config, track_source(config, reader, detections_stride=cached)


def score(name, split="tune"):
    t0 = time.time()
    reader = NvidiaMtmcReader(data_dir())
    (pa, sa), (pb, sb) = PAIRS[name]
    single = load_config(BASELINE_CONFIG)
    r = evaluate_fusion(
        reader,
        SPLITS[split],
        run(pa, sa, reader),
        (single, track_source(single, reader)),
        0.5,
        load_fusion(RECIPE),
        RESAMPLES,
        seed=0,
        compare_run=run(pb, sb, reader),
        truths=[TRUTH],
    )
    res = r.results[TRUTH]
    return {
        "split": split,
        "scenes": [s for s, _ in SPLITS[split]],
        "recipe": f"{RECIPE}, stitch off, from the stride-{CACHED_STRIDE} detection cache",
        "fusion": vars(load_fusion(RECIPE)),
        "pair": name,
        "first": {"passage": pa, "stride": sa},
        "second": {"passage": pb, "stride": sb},
        "truth": TRUTH,
        "n_resamples": RESAMPLES,
        "seed": 0,
        "interval": "paired block bootstrap, 30 s blocks, recentred percentile 95 %",
        "source_sha256": source_sha256(),
        "camera_tracks": {s.scene: s.camera_tracks for s in r.scenes},
        "fused_dwell_sha256": {
            s: hashlib.sha256(g.dwell.tobytes()).hexdigest() for s, g in r.fused.items()
        },
        "fused_passage_total": {s: float(g.passage.sum()) for s, g in r.fused.items()},
        "pooled": {
            "first": table(res.pooled[0]),
            "second": table(res.pooled[1]),
            "delta": table(res.delta_pooled),
        },
        "scenes_results": {
            x.scene: {
                "first": table(res.scenes[i][0]),
                "second": table(res.scenes[i][1]),
                "delta": table(res.delta_scenes[i]),
            }
            for i, x in enumerate(r.scenes)
        },
        "seconds": round(time.time() - t0),
    }


def line(r):
    p = r["pooled"]
    m = lambda k: p[k]["passage"]["mass_error"]  # noqa: E731
    t = lambda k: p[k]["passage"]["tvd"]  # noqa: E731
    d, dt = m("delta"), t("delta")
    return (
        f"{r['pair']}: passage mass err {m('first')['value']:.4f} vs {m('second')['value']:.4f} "
        f"delta {d['value']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]; passage TVD "
        f"{t('first')['value']:.4f} vs {t('second')['value']:.4f} delta {dt['value']:+.4f} "
        f"[{dt['lo']:+.4f}, {dt['hi']:+.4f}]; dwell delta TVD "
        f"{p['delta']['dwell']['tvd']['value']:+.4f} ({r['seconds']} s)"
    )


def dwell(split="tune"):
    """Per camera and stride, the sha256 of the dwell blocks with ids and with
    chord passage (they must match), and as an observation outside the rule
    the truth's passage mass on the joint coverage counted both ways."""
    reader = NvidiaMtmcReader(data_dir())
    out = {"cameras": {}, "truth_passage_full_joint": {}}
    for scene, anchor in SPLITS[split]:
        spec = GridSpec.covering(*reader.scene_extent(scene), cell_m=0.5)
        truth = reader.floor_truth(scene, anchor)
        first, last = int(truth.frames.min()), int(truth.frames.max())
        rate = reader.camera(scene, anchor).fps
        bf = round(BLOCK_S * rate)
        nb = last // bf + 1
        joint = np.zeros((spec.ny, spec.nx), dtype=bool)
        for c in map(str, reader.cameras(scene)):
            cam = reader.camera(scene, c)
            joint |= Homography(cam.world_to_image).sees(
                *spec.cell_centres(), cam.width, cam.height
            )
            for stride in (10, 2):
                config, source = run("ids", stride, reader)
                t = source.tracks(scene, c)
                keep = (t.frames >= first) & (t.frames <= last)
                ids, xs, ys, _ = recipe_floor_points(cam, t, config)
                args = (spec, ids[keep], t.frames[keep], xs[keep], ys[keep], cam.fps, bf, nb)
                h = {
                    mode: hashlib.sha256(block_layers(*args, mode).dwell.tobytes()).hexdigest()
                    for mode in ("ids", "chord")
                }
                out["cameras"][f"{scene}/{c}/s{stride}"] = h | {"equal": h["ids"] == h["chord"]}
        rows = (truth.track_ids, truth.frames, *truth.world.T, rate, bf, nb)
        mass = {
            m: float(block_layers(spec, *rows, m).passage.sum(0)[joint].sum())
            for m in ("ids", "chord")
        }
        out["truth_passage_full_joint"][scene] = mass | {"ratio": mass["chord"] / mass["ids"]}
        print(scene, out["truth_passage_full_joint"][scene], flush=True)
    out["dwell_byte_identical"] = all(v["equal"] for v in out["cameras"].values())
    return out


def write(out, report):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")


def read(name):
    return json.loads(Path(f"reports/chord-{name}-tune.json").read_text())


def decide():
    base, c10, c2, dw = (read(n) for n in ("baseline", "chord10", "chord2", "dwell"))
    mass = lambda r, k: r["pooled"][k]["passage"]["mass_error"]  # noqa: E731
    d = mass(c10, "delta")
    dtvd = c10["pooled"]["delta"]["passage"]["tvd"]
    s10, s2 = mass(c2, "first")["value"], mass(c2, "second")["value"]
    gap = abs(s10 - s2)
    same_baseline = mass(base, "first")["value"] == mass(c10, "second")["value"]
    dwell_ok = dw["dwell_byte_identical"] and all(
        c10["fused_dwell_sha256"][s] == base["fused_dwell_sha256"][s]
        for s in base["fused_dwell_sha256"]
    )
    effect = bool((d["hi"] < 0 or d["lo"] > 0) and d["value"] <= MARGIN)
    invariant = gap <= INVARIANCE
    guard = dtvd["hi"] <= GUARD
    out = {
        "rule": "accept: delta passage mass error full@joint (chord s10 - ids s10) CI excludes 0 "
        "and delta <= -.01; |mass error(chord s10) - mass error(chord s2)| <= .05; guards: "
        "delta passage TVD upper bound <= +.01, dwell byte-identical",
        "baseline_ids_s10": mass(base, "first"),
        "ids_s2": mass(base, "second"),
        "ids_stride_gap": abs(mass(base, "first")["value"] - mass(base, "second")["value"]),
        "chord_s10": mass(c10, "first"),
        "chord_s2": mass(c2, "second"),
        "delta_mass_error": d,
        "delta_passage_tvd": dtvd,
        "stride_gap_chord": gap,
        "delta_chord_s10_minus_s2": mass(c2, "delta"),
        "baseline_reproduced_in_pair": same_baseline,
        "effect": effect,
        "stride_invariance": invariant,
        "tvd_guard": guard,
        "dwell_byte_identical": dwell_ok,
        "accepted": bool(effect and invariant and guard and dwell_ok),
        "target_chord_s10_le_0.30": mass(c10, "first")["value"] <= TARGET,
    }
    write("reports/chord-decision-tune.json", out)
    for r in (base, c10, c2):
        print(line(r))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode in PAIRS:
        res = score(mode)
        print(line(res), flush=True)
        write(f"reports/chord-{mode}-tune.json", res)
    elif mode == "dwell":
        write("reports/chord-dwell-tune.json", dwell())
    elif mode == "decide":
        decide()
    else:
        raise SystemExit(__doc__)
