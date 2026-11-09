"""k=5 counted on coarser units (improvements plan, phase 7, H6), as
pre-registered in iterations.md, "Improvements plan, phase 7" (row H6).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 uv run --no-sync python scripts/k-zones-sweep.py

Hypothesis: the k-violations of phase 6 (H3: .194 per camera, .178 fused) come
from foot-point error that spreads one person over neighbouring 0.5 m cells.
Counting k on coarser units removes most of them.

Variants (raw track ids in all, no stitching):
- Z0: today. Display 0.5 m cells, k counted per 0.5 m cell.
- Z1: display 0.5 m cells, k counted on the 1 m cell (2x2 block) that holds it.
- Z2: display and k on 1 m cells.
- Z3: display and k on 2 m zones.

Recipe and slot: as scripts/k-chains-sweep.py (H3, branch feat/39-k-on-chains).
configs/e2a-weights.yaml at stride 10, replayed from the stride-2 detection
cache (track_source(..., detections_stride=2)), no stitch in the tracks, W1
fusion. Tune 071-075, every calibrated camera; sealed 076-080 is not scored.
One slot per camera: the tracked rows in the truth frames [first, last] of the
scene. CPU only.

Choices fixed here before the first run (not in the pre-registration):
- Units. Everything is computed on the scene grid at 0.5 m
  (GridSpec.covering snaps the extent to 2 m, so the 1 m and 2 m units are
  exact 2x2 and 4x4 blocks of it, aligned at the origin). A unit's dwell is
  the sum of its cells; its distinct ids are the distinct ids over its cells.
- Shown display unit: its k unit passes (distinct ids >= 5; fused: the fused
  count >= 5) and its predicted raw dwell is > 0. For Z0 this is H3's "shown".
  Z1 judged on its 1 m unit: a 1 m block is shown when it passes k and its
  predicted dwell is > 0.
- Violation: a shown unit whose true distinct people (primary: visible, as
  H3; secondary: full) in the slot are < 5, on the same unit. Rate: violating
  over shown units, pooled sum over sum over the bootstrap units (cameras;
  fused: scenes).
- Fused (`run --site`): k after fusion. Each camera's distinct ids per k unit
  are fused with W1 weights (fusion_weights of the cameras' whole-slot raw
  0.5 m grids, as H3), averaged to the unit: a unit's weight for a camera is
  the sum of its 0.5 m weights over the unit's cells, renormalised over the
  cameras (for 0.5 m units: the weights themselves). The fused dwell of a
  display unit is the cameras' unit dwell fused with the same unit weights.
  Truth: the people at least one of the scene's cameras has a visible box of.
- Hidden mass (guard): true dwell (primary truth) in the 0.5 m cells, on the
  coverage (per camera: its coverage; fused: the joint coverage), of the
  display units with predicted dwell > 0 that k hides, over the true dwell on
  the coverage; pooled sum over sum. Guard: delta against Z0 (point) <= +.10,
  on both surfaces.
- TVD guard: dwell TVD at full@joint of the fused shown map after k, on the
  evaluation's 0.5 m grid (compare_layer), each display unit's dwell spread
  evenly over its cells as the page draws it. Per scene, then the mean over
  the 5 scenes. Guard: delta against Z0 (point) <= +.02. Reported too: the
  same TVD with prediction and truth summed to the display unit (own
  resolution), and per camera at full@coverage.
- CI: bootstrap over units (cameras; fused and TVD: scenes), 1000 resamples,
  seed 0 (numpy default_rng), percentile 95 %, paired with Z0 on the same
  resample.
- Target: pooled violation rate <= .05 per camera and fused. Z1 must hold it
  on both its 0.5 m cell and its 1 m unit.
- Selection: among the variants that reach the target on both surfaces and
  hold both guards, the finest display unit (Z1, then Z2, then Z3). If none:
  among the variants that hold both guards, the one with the lowest larger
  rate of the two surfaces, kept only if its delta CI against Z0 excludes 0 on
  both surfaces.
- Live layer (reported, no rule): at instants every 60 s from first + 120 s
  to last, per camera, a display unit is shown where its k unit had >= 5
  distinct ids in the last 120 s and at least one id was seen in the display
  unit in that window; truth: the visible people in that unit in the same
  window. Fused: the unit-weighted counts as above, present where the fused
  display-unit count is > 0. Per camera CI over cameras.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from space_usage_heatmaps.cli import data_dir, load_config, load_fusion, source_sha256, track_source
from space_usage_heatmaps.evaluation.application.evaluate_oracle import recipe_floor_points
from space_usage_heatmaps.evaluation.domain.metrics import compare_layer
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.fusion import CameraGrid, fusion_weights
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec
from space_usage_heatmaps.occupancy.domain.live import RECENT_S

RECIPE = "configs/e2a-weights.yaml"
STRIDE, CACHED_STRIDE = 10, 2
CELL_M = 0.5
K = 5
RESAMPLES, SEED = 1000, 0
TARGET, HIDDEN_GUARD, TVD_GUARD = 0.05, 0.10, 0.02
LIVE_STEP_S = 60
# name: (display factor, k factor) in 0.5 m cells
VARIANTS = {"Z0": (1, 1), "Z1": (1, 2), "Z2": (2, 2), "Z3": (4, 4)}
# what each variant is judged on: (name, unit factor); Z1 twice
JUDGED = {
    "Z0": (("Z0", 1),),
    "Z1": (("Z1", 1), ("Z1@1m", 2)),
    "Z2": (("Z2", 2),),
    "Z3": (("Z3", 4),),
}
ARMS = ("Z0", "Z1", "Z1@1m", "Z2", "Z3")
TRUTHS = ("visible", "full")
OUT = Path("reports/k-zones-tune.json")


def run_config():
    return replace(load_config(RECIPE), stride=STRIDE, stitch=None)


def block_sum(a, f):
    """Sum f x f blocks of the last two axes (the grid snaps to 2 m, so they divide)."""
    if f == 1:
        return np.asarray(a, float)
    *lead, ny, nx = a.shape
    return np.asarray(a, float).reshape(*lead, ny // f, f, nx // f, f).sum(axis=(-3, -1))


def up(a, f):
    """Each unit's value on its f x f cells."""
    return a if f == 1 else np.repeat(np.repeat(a, f, axis=-2), f, axis=-1)


def distinct(spec, ids, xs, ys, f):
    """Distinct ids per unit of f x f cells, shape (ny/f, nx/f)."""
    cells, inside = spec.cell_index(xs, ys)
    ids = np.asarray(ids)[inside]
    r, c = cells // spec.nx // f, cells % spec.nx // f
    nbx = spec.nx // f
    out = np.zeros((spec.ny // f) * nbx)
    if len(ids):
        pairs = np.unique(np.stack([r * nbx + c, ids]), axis=1)
        np.add.at(out, pairs[0], 1.0)
    return out.reshape(spec.ny // f, nbx)


def unit_weights(w, f):
    """Per camera, the sum of its 0.5 m weights over each unit, renormalised over cameras."""
    s = block_sum(w, f)
    total = s.sum(axis=0)
    return np.divide(s, total, out=np.zeros_like(s), where=total > 0)


def spread(unit_dwell, f):
    """A unit's dwell spread evenly over its cells, as the page draws it."""
    return up(np.asarray(unit_dwell, float), f) / (f * f)


def judge(shown, truth_people):
    return {"shown": int(shown.sum()), "violations": int((shown & (truth_people < K)).sum())}


def camera_arms(spec, ids, xs, ys, dwell, truths, coverage):
    """Per arm and truth: shown, violations, hidden and total true mass of one slot."""
    out = {}
    for name, (d, kf) in VARIANTS.items():
        kpass = up(distinct(spec, ids, xs, ys, kf) >= K, kf)  # 0.5 m cells
        for arm, f in JUDGED[name]:
            unit_pass = block_sum(kpass, f) > 0  # k units are never finer than judged units
            shown = unit_pass & (block_sum(dwell, f) > 0)
            out[arm] = {}
            for tr, (people_of, tdwell) in truths.items():
                entry = judge(shown, people_of(f))
                if arm == name:  # hidden mass on the display unit
                    hidden_units = (block_sum(dwell, d) > 0) & ~(block_sum(kpass, d) > 0)
                    hidden = up(hidden_units, d) & coverage
                    entry["hidden_mass"] = float(tdwell[hidden].sum())
                    entry["mass"] = float(tdwell[coverage].sum())
                out[arm][tr] = entry
    return out


def measure_scene(scene):
    t0 = time.time()
    reader = NvidiaMtmcReader(data_dir())
    config = run_config()
    source = track_source(config, reader, detections_stride=CACHED_STRIDE)
    fusion = load_fusion(RECIPE)
    spec = GridSpec.covering(*reader.scene_extent(scene), cell_m=CELL_M)
    cameras = reader.cameras(scene)
    truth_all = reader.floor_truth(scene, cameras[0])
    first, last = int(truth_all.frames.min()), int(truth_all.frames.max())
    seen_by_any = np.zeros(len(truth_all.track_ids), dtype=bool)
    per_camera, cams, rates = [], [], set()
    for c in cameras:
        cam = reader.camera(scene, c)
        fps = cam.fps
        rates.add(fps)
        t = source.tracks(scene, c)
        ids_all, xs_all, ys_all, _ = recipe_floor_points(cam, t, config)
        keep = (t.frames >= first) & (t.frames <= last)
        order = np.argsort(t.frames[keep], kind="stable")
        ids = np.asarray(ids_all)[keep][order]
        frames = np.asarray(t.frames)[keep][order]
        xs, ys = xs_all[keep][order], ys_all[keep][order]
        coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
        truth = reader.floor_truth(scene, c)
        seen_by_any |= truth.visible
        v = truth.visible
        vis = (truth.track_ids[v], *truth.world[v].T)
        full = (truth.track_ids, *truth.world.T)
        truths = {
            "visible": (
                lambda f, s=vis: distinct(spec, *s, f),
                Grid.from_samples(spec, *vis, fps).dwell,
            ),
            "full": (
                lambda f, s=full: distinct(spec, *s, f),
                Grid.from_samples(spec, *full, fps).dwell,
            ),
        }
        raw = Grid.from_samples(spec, ids, xs, ys, fps)
        unit = {
            "scene": scene,
            "camera": c,
            "rows": int(len(ids)),
            "raw_ids": int(len(np.unique(ids))),
        }
        unit |= camera_arms(spec, ids, xs, ys, raw.dwell, truths, coverage)
        full_dwell = truths["full"][1]
        unit["tvd_coverage"] = {}
        for name, (d, kf) in VARIANTS.items():
            kpass = up(distinct(spec, ids, xs, ys, kf) >= K, kf)
            shown_map = spread(block_sum(np.where(kpass, raw.dwell, 0.0), d), d)
            unit["tvd_coverage"][name] = tvd(shown_map, full_dwell, coverage, d)
        per_camera.append(unit)
        cams.append((c, fps, ids, frames, xs, ys, raw, coverage, truth))
    if len(rates) != 1:
        raise ValueError(f"scene {scene}: cameras at different frame rates {sorted(rates)}")
    (fps,) = rates

    # Fused slot: k after fusion, as run --site.
    grids = [CameraGrid(c, raw, cov) for c, _, _, _, _, _, raw, cov, _ in cams]
    w = fusion_weights(grids, fusion)
    joint = np.logical_or.reduce([g.coverage for g in grids])
    va = seen_by_any
    vis = (truth_all.track_ids[va], *truth_all.world[va].T)
    full = (truth_all.track_ids, *truth_all.world.T)
    ftruth = {
        "visible": (
            lambda f, s=vis: distinct(spec, *s, f),
            Grid.from_samples(spec, *vis, fps).dwell,
        ),
        "full": (
            lambda f, s=full: distinct(spec, *s, f),
            Grid.from_samples(spec, *full, fps).dwell,
        ),
    }
    fused = {"scene": scene, "cameras": len(cams), "tvd_joint": {}, "tvd_joint_own": {}}
    for name, (d, kf) in VARIANTS.items():
        wk = unit_weights(w, kf)
        kcount = sum(
            wk[i] * distinct(spec, ids, xs, ys, kf)
            for i, (_, _, ids, _, xs, ys, _, _, _) in enumerate(cams)
        )
        kpass = up(kcount >= K, kf)
        wd = unit_weights(w, d)
        fdwell_d = sum(wd[i] * block_sum(c[6].dwell, d) for i, c in enumerate(cams))
        fdwell_cell = spread(fdwell_d, d)  # raw fused dwell on the cells
        shown_map = spread(np.where(block_sum(kpass, d) > 0, fdwell_d, 0.0), d)
        fused["tvd_joint"][name] = tvd(shown_map, ftruth["full"][1], joint, 1)
        fused["tvd_joint_own"][name] = tvd(shown_map, ftruth["full"][1], joint, d)
        for arm, f in JUDGED[name]:
            shown = (block_sum(kpass, f) > 0) & (block_sum(fdwell_cell, f) > 0)
            fused[arm] = {}
            for tr, (people_of, tdwell) in ftruth.items():
                entry = judge(shown, people_of(f))
                if arm == name:
                    hidden_units = (fdwell_d > 0) & ~(block_sum(kpass, d) > 0)
                    hidden = up(hidden_units, d) & joint
                    entry["hidden_mass"] = float(tdwell[hidden].sum())
                    entry["mass"] = float(tdwell[joint].sum())
                fused[arm][tr] = entry

    # Live layer: 120 s windows every 60 s.
    window = round(RECENT_S * fps)
    instants = list(range(first + window, last + 1, round(LIVE_STEP_S * fps)))
    t_ids, t_frames, t_world = truth_all.track_ids, truth_all.frames, truth_all.world
    live_cam = {}
    live_fused = {a: {"shown": 0, "violations": 0} for a in ARMS}
    for f0 in instants:
        in_t = (t_frames >= f0 - window) & (t_frames <= f0)
        any_rows = in_t & seen_by_any
        counts_by = {}  # (camera index, factor) -> distinct ids in the window
        for i, (c, _, ids, frames, xs, ys, _, _, truth) in enumerate(cams):
            rows = (frames >= f0 - window) & (frames <= f0)
            tv = in_t & truth.visible
            u = live_cam.setdefault(
                (scene, c), {a: {"t": {"shown": 0, "violations": 0}} for a in ARMS}
            )
            for f in (1, 2, 4):
                counts_by[i, f] = distinct(spec, ids[rows], xs[rows], ys[rows], f)
            for name, (d, kf) in VARIANTS.items():
                kpass = up(counts_by[i, kf] >= K, kf)
                present = up(counts_by[i, d] > 0, d)
                for arm, f in JUDGED[name]:
                    shown = block_sum(kpass & present, f) > 0
                    tp = distinct(spec, t_ids[tv], *t_world[tv].T, f)
                    e = judge(shown, tp)
                    for key in e:
                        u[arm]["t"][key] += e[key]
        for name, (d, kf) in VARIANTS.items():
            wk, wd = unit_weights(w, kf), unit_weights(w, d)
            kpass = up(sum(wk[i] * counts_by[i, kf] for i in range(len(cams))) >= K, kf)
            present = up(sum(wd[i] * counts_by[i, d] for i in range(len(cams))) > 0, d)
            for arm, f in JUDGED[name]:
                shown = block_sum(kpass & present, f) > 0
                e = judge(shown, distinct(spec, t_ids[any_rows], *t_world[any_rows].T, f))
                for key in e:
                    live_fused[arm][key] += e[key]
    return {
        "scene": scene,
        "frames": [first, last],
        "fps": fps,
        "cameras": per_camera,
        "fused": fused,
        "live_cameras": [{"scene": s, "camera": c, **v} for (s, c), v in live_cam.items()],
        "live_fused": live_fused | {"instants": len(instants)},
        "seconds": round(time.time() - t0),
    }


def tvd(pred, truth, mask, f):
    """Dwell TVD on the mask, with prediction and truth summed to units of f x f cells."""
    p, t = np.where(mask, pred, 0.0), np.where(mask, truth, 0.0)
    if f > 1:
        p, t = block_sum(p, f), block_sum(t, f)
        mask = np.ones(p.shape, bool)
    return float(compare_layer(p, t, mask, CELL_M * f, zones=False).tvd)


def pooled(units, arm, truth, key="violations", total="shown"):
    num = sum(u[arm][truth][key] for u in units)
    den = sum(u[arm][truth][total] for u in units)
    return num / den if den else 0.0


def with_ci(point, draws):
    d = np.array(draws)
    return {"value": point, "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def bootstrap(units, truth):
    """Pooled violation rate per arm and hidden share per variant, paired deltas against Z0."""
    rng = np.random.default_rng(SEED)
    n = len(units)

    def stats(sample):
        rate = {a: pooled(sample, a, truth) for a in ARMS}
        hidden = {v: pooled(sample, v, truth, "hidden_mass", "mass") for v in VARIANTS}
        return {
            "violation_rate": rate | {f"delta_{a}": rate[a] - rate["Z0"] for a in ARMS[1:]},
            "hidden_share": hidden
            | {f"delta_{v}": hidden[v] - hidden["Z0"] for v in VARIANTS if v != "Z0"},
        }

    point = stats(units)
    draws = [stats([units[i] for i in rng.integers(0, n, n)]) for _ in range(RESAMPLES)]
    out = {m: {k: with_ci(v, [x[m][k] for x in draws]) for k, v in point[m].items()} for m in point}
    out["shown"] = {a: sum(u[a][truth]["shown"] for u in units) for a in ARMS}
    out["violations"] = {a: sum(u[a][truth]["violations"] for u in units) for a in ARMS}
    return out


def tvd_bootstrap(values):
    """Mean over units of a per-unit TVD per variant, paired deltas against Z0."""
    rng = np.random.default_rng(SEED)
    n = len(values)

    def stats(sample):
        m = {v: float(np.mean([s[v] for s in sample])) for v in VARIANTS}
        return m | {f"delta_{v}": m[v] - m["Z0"] for v in VARIANTS if v != "Z0"}

    point = stats(values)
    draws = [stats([values[i] for i in rng.integers(0, n, n)]) for _ in range(RESAMPLES)]
    return {k: with_ci(v, [x[k] for x in draws]) for k, v in point.items()}


def live_bootstrap(units):
    rng = np.random.default_rng(SEED)
    n = len(units)
    out = {}
    for a in ARMS:
        draws = [
            pooled([units[i] for i in rng.integers(0, n, n)], a, "t") for _ in range(RESAMPLES)
        ]
        out[a] = with_ci(pooled(units, a, "t"), draws) | {
            "shown": sum(u[a]["t"]["shown"] for u in units),
            "violations": sum(u[a]["t"]["violations"] for u in units),
        }
    return out


def decide(report):
    p = TRUTHS[0]
    cam, fus = report["per_camera"][p], report["fused"][p]
    tvd_j = report["tvd_full_joint"]
    rules = {}
    for name in VARIANTS:
        if name == "Z0":
            continue
        arms = [a for a, _ in JUDGED[name]]
        checks = {
            "per_camera_le_target": all(cam["violation_rate"][a]["value"] <= TARGET for a in arms),
            "fused_le_target": all(fus["violation_rate"][a]["value"] <= TARGET for a in arms),
            "hidden_guard_per_camera": cam["hidden_share"][f"delta_{name}"]["value"]
            <= HIDDEN_GUARD,
            "hidden_guard_fused": fus["hidden_share"][f"delta_{name}"]["value"] <= HIDDEN_GUARD,
            "tvd_guard": tvd_j[f"delta_{name}"]["value"] <= TVD_GUARD,
            "delta_ci_excludes_0": all(
                s["violation_rate"][f"delta_{a}"]["hi"] < 0 for s in (cam, fus) for a in arms
            ),
        }
        guards = (
            checks["hidden_guard_per_camera"]
            and checks["hidden_guard_fused"]
            and checks["tvd_guard"]
        )
        checks["passes"] = guards and checks["per_camera_le_target"] and checks["fused_le_target"]
        checks["guards"] = guards
        checks["worst_rate"] = max(
            s["violation_rate"][a]["value"] for s in (cam, fus) for a in arms
        )
        rules[name] = checks
    passing = [v for v in ("Z1", "Z2", "Z3") if rules[v]["passes"]]
    if passing:
        chosen, why = passing[0], "finest display unit among the variants that pass"
    else:
        guarded = sorted(
            (v for v in rules if rules[v]["guards"]), key=lambda v: rules[v]["worst_rate"]
        )
        if guarded and rules[guarded[0]]["delta_ci_excludes_0"]:
            chosen, why = guarded[0], "none passes; best guard-holding variant, CI excludes Z0"
        else:
            chosen, why = "Z0", "none passes and no guard-holding variant has a CI that excludes Z0"
    return {"variants": rules, "chosen": chosen, "why": why}


def main():
    t0 = time.time()
    scenes = [s for s, _ in SPLITS["tune"]]
    with ProcessPoolExecutor(int(os.environ.get("SWEEP_WORKERS", 3))) as pool:
        results = list(pool.map(measure_scene, scenes))
    cams = [u for r in results for u in r["cameras"]]
    fused = [r["fused"] for r in results]
    live_units = [{a: u[a] for a in ARMS} for r in results for u in r["live_cameras"]]
    report = {
        "hypothesis": "H6: k=5 counted on coarser units (Z1 1 m k / 0.5 m display, Z2 1 m, Z3 2 m)",
        "recipe": f"{RECIPE} at stride {STRIDE} from the stride-{CACHED_STRIDE} detection cache, "
        "no stitch in the tracks, W1 fusion, raw ids",
        "variants": {
            k: {"display_m": d * CELL_M, "k_m": kf * CELL_M} for k, (d, kf) in VARIANTS.items()
        },
        "k": K,
        "split": "tune",
        "scenes": scenes,
        "n_resamples": RESAMPLES,
        "seed": SEED,
        "interval": "bootstrap over units (cameras; fused, TVD: scenes), percentile 95 %, "
        "paired with Z0",
        "primary_truth": TRUTHS[0],
        "source_sha256": source_sha256(),
        "per_camera": {tr: bootstrap(cams, tr) for tr in TRUTHS},
        "fused": {tr: bootstrap(fused, tr) for tr in TRUTHS},
        "tvd_full_joint": tvd_bootstrap([f["tvd_joint"] for f in fused]),
        "tvd_full_joint_own_unit": tvd_bootstrap([f["tvd_joint_own"] for f in fused]),
        "tvd_full_coverage_per_camera": tvd_bootstrap([u["tvd_coverage"] for u in cams]),
        "per_scene": {
            r["scene"]: {
                "cameras": len(r["cameras"]),
                "per_camera": {a: pooled(r["cameras"], a, "visible") for a in ARMS},
                "fused": {a: pooled([r["fused"]], a, "visible") for a in ARMS},
                "seconds": r["seconds"],
            }
            for r in results
        },
        "live": {
            "per_camera": live_bootstrap(live_units),
            "fused": {
                a: {
                    "value": sum(r["live_fused"][a]["violations"] for r in results)
                    / max(1, sum(r["live_fused"][a]["shown"] for r in results)),
                    "shown": sum(r["live_fused"][a]["shown"] for r in results),
                }
                for a in ARMS
            },
        },
        "cameras": cams,
        "fused_scenes": fused,
        "seconds": round(time.time() - t0),
    }
    report["rule"] = decide(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    p = TRUTHS[0]
    for surface in ("per_camera", "fused"):
        b = report[surface][p]
        for a in ARMS:
            v = b["violation_rate"][a]
            line = f"{surface} {a}: rate {v['value']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]"
            line += f" shown {b['shown'][a]}"
            if a != "Z0":
                dv = b["violation_rate"][f"delta_{a}"]
                line += f" delta {dv['value']:+.4f} [{dv['lo']:+.4f}, {dv['hi']:+.4f}]"
            if a in VARIANTS:
                h = b["hidden_share"][a]
                line += f"; hidden {h['value']:.4f}"
            print(line)
    for key in ("tvd_full_joint", "tvd_full_joint_own_unit", "tvd_full_coverage_per_camera"):
        print(key, {k: round(v["value"], 4) for k, v in report[key].items()})
    print(
        "live per camera",
        {a: round(v["value"], 4) for a, v in report["live"]["per_camera"].items()},
    )
    print("live fused", {a: round(v["value"], 4) for a, v in report["live"]["fused"].items()})
    print(json.dumps(report["rule"], indent=2))
    print(f"{OUT}, {report['seconds']} s")


if __name__ == "__main__":
    main()
