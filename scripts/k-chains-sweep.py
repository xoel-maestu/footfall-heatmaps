"""k=5 on stitched chains (improvements plan phase 6, H3), as pre-registered in
iterations.md, "Improvements plan, phase 6" (row H3).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 uv run --no-sync python scripts/k-chains-sweep.py

Hypothesis: k=5 counted on raw track ids lets cells through that fewer than 5
real people crossed, because tracks fragment. Counting k on the slot's
stitched chains (G10 R1, occupancy.domain.stitching.K_STITCH) instead lowers
the k-violation rate to <= 1 %.

Recipe: configs/e2a-weights.yaml (the product recipe; same per-camera recipe as
configs/e2a-record.yaml) at stride 10, replayed from the stride-2 TensorRT
detection cache (track_source(..., detections_stride=2)), stitching off in the
tracks, W1 fusion. Tune scenes 071-075, every calibrated camera. Sealed 076-080
is not scored. CPU only.

Both arms go through the product's own code: occupancy.domain.slots.SlotAccumulator
with stitch=None (raw ids, today) or stitch=K_STITCH (chains), fed every
tracked row of the slot (filled frames included, as the recorder gets them).
A self-check compares the online chains with the batch stitch() of evaluation.

Choices fixed here before the first run (not in the pre-registration):
- Slot: the videos are about 800 s, so each camera has one (incomplete) 15-min
  slot. Its rows are the tracked rows in the truth frames [first, last] of the
  scene (the evaluation's window), on the scene grid at 0.5 m.
- Shown cell: the k count (raw ids or chains) >= 5 in the slot. Per camera this
  is `run`; fused it is `run --site`: the cameras' k counts fused with the W1
  weights of the cameras' whole-slot raw dwell (fusion_weights; the product
  freezes the same formula from a calibration window), then >= 5.
- True distinct people of a cell (primary): the people the camera has a visible
  box of (floor_truth visible), distinct per cell over the slot; fused: the
  people at least one of the scene's cameras has a visible box of. Reason: the
  shown value is made only from people some camera saw, so those are the
  people k protects. Secondary, reported only: every person (full truth).
- k-violation rate: violating shown cells / shown cells, pooled over the units
  (sum over sum). Per camera: units are (scene, camera); fused: units are scenes.
- Hidden mass (guard): true dwell (same truth) in cells with raw predicted
  dwell > 0 that k hides, over the true dwell on the coverage (per camera: the
  camera's coverage; fused: the joint coverage), pooled sum over sum.
- CI: bootstrap over units (resample cameras with replacement; fused: scenes),
  1000 resamples, seed 0 (numpy default_rng), percentile 95 %, paired
  (chains - raw ids on the same resample).
- Rule, per surface (per camera -> `run`; fused -> `run --site`), primary truth:
  accept when the delta violation rate CI upper bound < 0, the chains pooled
  rate <= .01, and the pooled delta hidden mass (point value) <= +.05. A surface
  that passes gets chains as its default for persisted slots.
- Live layer (reported, no rule): the product keeps raw ids live. Per camera
  and fused, at instants every 60 s from first + 120 s to last, a cell is shown
  where the distinct ids seen in it in the last 120 s reach 5 (LiveLayer); the
  truth is the visible people in that cell in the same 120 s. Chains computed
  over the whole slot are reported next to it as a diagnostic only (a live
  stitch cannot read the future).
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
from space_usage_heatmaps.evaluation.domain.splits import SPLITS
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.fusion import CameraGrid, fuse_layers, fusion_weights
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec
from space_usage_heatmaps.occupancy.domain.live import RECENT_S
from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, SlotAccumulator
from space_usage_heatmaps.occupancy.domain.stitching import K_STITCH, stitch

RECIPE = "configs/e2a-weights.yaml"
STRIDE, CACHED_STRIDE = 10, 2
CELL_M = 0.5
RESAMPLES, SEED = 1000, 0
TARGET, GUARD = 0.01, 0.05
LIVE_STEP_S = 60
ARMS = ("ids", "chains")
TRUTHS = ("visible", "full")  # primary first
OUT = Path("reports/k-chains-tune.json")


def run_config():
    return replace(load_config(RECIPE), stride=STRIDE, stitch=None)


def distinct_per_cell(spec, ids, xs, ys):
    """Distinct ids per cell (passage of Grid.from_samples), and the dwell rows' cells."""
    return Grid.from_samples(spec, ids, xs, ys, 1.0).passage


def slot_counts(spec, ids, frames, xy, fps, arm):
    """The product's slot for one arm: (raw dwell, k count per cell, chains or None)."""
    acc = SlotAccumulator(spec, 0, stitch=K_STITCH if arm == "chains" else None)
    cells, inside = spec.cell_index(xy[:, 0], xy[:, 1])
    acc.add(ids[inside], cells, 1 / fps)
    acc.follow(ids, frames, xy, fps)
    dwell, passage = acc.raw()
    people, _ = acc.people()
    chain = acc._chain_of() if arm == "chains" else None
    return dwell.copy(), (passage if people is None else people).copy(), chain


def counts(shown, truth_people, hidden, truth_dwell, mask):
    """Scalars of one unit: shown cells, violations, hidden and total true mass."""
    return {
        "shown": int(shown.sum()),
        "violations": int((shown & (truth_people < K_MIN_PEOPLE)).sum()),
        "hidden_mass": float(truth_dwell[hidden & mask].sum()),
        "mass": float(truth_dwell[mask].sum()),
    }


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
    per_camera, grids, people_by_arm, live_rows = [], [], {a: [] for a in ARMS}, []
    rates = set()
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
        xy = np.column_stack([xs_all[keep][order], ys_all[keep][order]])
        coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
        truth = reader.floor_truth(scene, c)
        seen_by_any |= truth.visible
        truth_grids = {
            "visible": Grid.from_samples(spec, truth.track_ids[truth.visible],
                                         *truth.world[truth.visible].T, fps),
            "full": Grid.from_samples(spec, truth.track_ids, *truth.world.T, fps),
        }  # fmt: skip
        unit = {"scene": scene, "camera": c, "rows": int(len(ids)),
                "raw_ids": int(len(np.unique(ids)))}  # fmt: skip
        chain = None
        for arm in ARMS:
            dwell, people, ch = slot_counts(spec, ids, frames, xy, fps, arm)
            if arm == "chains":
                chain = ch
                batch = stitch(ids, frames, xy, fps, K_STITCH)
                unit["chains"] = int(len(set(ch.values())))
                unit["online_equals_batch"] = bool(
                    np.array_equal(np.array([ch[i] for i in ids.tolist()]), batch)
                )
            else:
                grids.append(CameraGrid(c, Grid(spec, dwell, people, 0), coverage))
            people_by_arm[arm].append(people)
            shown = people >= K_MIN_PEOPLE
            hidden = (dwell > 0) & ~shown
            unit[arm] = {
                tr: counts(shown, g.passage, hidden, g.dwell, coverage)
                for tr, g in truth_grids.items()
            }
        per_camera.append(unit)
        live_rows.append((c, fps, ids, frames, xy, chain, truth))
    if len(rates) != 1:
        raise ValueError(f"scene {scene}: cameras at different frame rates {sorted(rates)}")
    (fps,) = rates

    # Fused slot: k after fusion, as run --site.
    weights = fusion_weights(grids, fusion)
    joint = np.logical_or.reduce([g.coverage for g in grids])
    fused_dwell = fuse_layers(weights, [g.grid.dwell for g in grids])
    vis = truth_all.track_ids[seen_by_any], *truth_all.world[seen_by_any].T
    fused_truth = {
        "visible": Grid.from_samples(spec, *vis, fps),
        "full": Grid.from_samples(spec, truth_all.track_ids, *truth_all.world.T, fps),
    }
    fused = {"scene": scene, "cameras": len(cameras)}
    for arm in ARMS:
        people = fuse_layers(weights, people_by_arm[arm])
        shown = people >= K_MIN_PEOPLE
        hidden = (fused_dwell > 0) & ~shown
        fused[arm] = {
            tr: counts(shown, g.passage, hidden, g.dwell, joint) for tr, g in fused_truth.items()
        }

    # Live layer (raw ids in the product; chains over the slot as a diagnostic).
    window = round(RECENT_S * fps)
    instants = list(range(first + window, last + 1, round(LIVE_STEP_S * fps)))
    live_cam, live_fused = [], {a: {"shown": 0, "violations": 0} for a in ARMS}
    t_ids, t_frames, t_world = truth_all.track_ids, truth_all.frames, truth_all.world
    for f in instants:
        in_t = (t_frames >= f - window) & (t_frames <= f)
        fused_people = {a: [] for a in ARMS}
        any_rows = in_t & seen_by_any
        fused_truth_people = distinct_per_cell(spec, t_ids[any_rows], *t_world[any_rows].T)
        for c, _, ids, frames, xy, chain, truth in live_rows:
            rows = (frames >= f - window) & (frames <= f)
            tv = in_t & truth.visible
            truth_people = distinct_per_cell(spec, t_ids[tv], *t_world[tv].T)
            entry = {"camera": c, "instant": int(f)}
            for arm in ARMS:
                lid = ids[rows]
                if arm == "chains":
                    lid = np.array([chain[i] for i in lid.tolist()], dtype=ids.dtype)
                people = distinct_per_cell(spec, lid, xy[rows, 0], xy[rows, 1])
                fused_people[arm].append(people)
                shown = people >= K_MIN_PEOPLE
                entry[arm] = {
                    "shown": int(shown.sum()),
                    "violations": int((shown & (truth_people < K_MIN_PEOPLE)).sum()),
                }
            live_cam.append(entry)
        for arm in ARMS:
            shown = fuse_layers(weights, fused_people[arm]) >= K_MIN_PEOPLE
            live_fused[arm]["shown"] += int(shown.sum())
            live_fused[arm]["violations"] += int(
                (shown & (fused_truth_people < K_MIN_PEOPLE)).sum()
            )
    return {
        "scene": scene,
        "frames": [first, last],
        "fps": fps,
        "cameras": per_camera,
        "fused": fused,
        "live_cameras": live_cam,
        "live_fused": live_fused | {"instants": len(instants)},
        "seconds": round(time.time() - t0),
    }


def pooled(units, arm, truth, key="violations", total="shown"):
    num = sum(u[arm][truth][key] for u in units)
    den = sum(u[arm][truth][total] for u in units)
    return num / den if den else 0.0


def bootstrap(units, truth):
    """Point values and paired percentile CIs of the pooled violation rate and
    hidden-mass share, resampling units with replacement."""
    rng = np.random.default_rng(SEED)
    n = len(units)

    def stats(sample):
        out = {}
        for metric, key, total in (("violation_rate", "violations", "shown"),
                                   ("hidden_share", "hidden_mass", "mass")):  # fmt: skip
            v = {a: pooled(sample, a, truth, key, total) for a in ARMS}
            out[metric] = v | {"delta": v["chains"] - v["ids"]}
        return out

    point = stats(units)
    draws = [stats([units[i] for i in rng.integers(0, n, n)]) for _ in range(RESAMPLES)]
    for metric, v in point.items():
        for name in (*ARMS, "delta"):
            d = np.array([x[metric][name] for x in draws])
            v[name] = {"value": v[name], "lo": float(np.percentile(d, 2.5)),
                       "hi": float(np.percentile(d, 97.5))}  # fmt: skip
    point["shown"] = {a: sum(u[a][truth]["shown"] for u in units) for a in ARMS}
    point["violations"] = {a: sum(u[a][truth]["violations"] for u in units) for a in ARMS}
    return point


def rule(b):
    v, h = b["violation_rate"], b["hidden_share"]
    checks = {
        "falls_ci_excludes_0": v["delta"]["hi"] < 0,
        "chains_pooled_le_target": v["chains"]["value"] <= TARGET,
        "guard_hidden_delta_le": h["delta"]["value"] <= GUARD,
    }
    return checks | {"accept": all(checks.values())}


def live_rate(entries):
    units = [{a: {"t": e[a]} for a in ARMS} for e in entries]
    return {a: pooled(units, a, "t") for a in ARMS} | {
        "shown": {a: sum(e[a]["shown"] for e in entries) for a in ARMS},
        "violations": {a: sum(e[a]["violations"] for e in entries) for a in ARMS},
    }


def live_bootstrap(scenes):
    """Per-camera live rate (raw ids), CI over cameras."""
    by_camera = {}
    for s in scenes:
        for e in s["live_cameras"]:
            empty = {a: {"t": {"shown": 0, "violations": 0}} for a in ARMS}
            u = by_camera.setdefault((s["scene"], e["camera"]), empty)
            for a in ARMS:
                for k in ("shown", "violations"):
                    u[a]["t"][k] += e[a][k]
    units = list(by_camera.values())
    rng = np.random.default_rng(SEED)
    out = {}
    for a in ARMS:
        draws = [pooled([units[i] for i in rng.integers(0, len(units), len(units))], a, "t")
                 for _ in range(RESAMPLES)]  # fmt: skip
        out[a] = {"value": pooled(units, a, "t"), "lo": float(np.percentile(draws, 2.5)),
                  "hi": float(np.percentile(draws, 97.5))}  # fmt: skip
    return out


def main():
    t0 = time.time()
    scenes = [s for s, _ in SPLITS["tune"]]
    workers = int(os.environ.get("SWEEP_WORKERS", 3))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(measure_scene, scenes))
    cams = [u for r in results for u in r["cameras"]]
    fused = [r["fused"] for r in results]
    report = {
        "hypothesis": "H3: k=5 on stitched chains (G10 R1) instead of raw track ids",
        "recipe": f"{RECIPE} at stride {STRIDE} from the stride-{CACHED_STRIDE} detection cache, "
        "no stitch in the tracks, W1 fusion",
        "stitch": vars(K_STITCH),
        "k": K_MIN_PEOPLE,
        "split": "tune",
        "scenes": scenes,
        "n_resamples": RESAMPLES,
        "seed": SEED,
        "interval": "bootstrap over units (cameras; fused: scenes), percentile 95 %, paired",
        "primary_truth": TRUTHS[0],
        "source_sha256": source_sha256(),
        "online_equals_batch": all(u["online_equals_batch"] for u in cams),
        "per_camera": {tr: bootstrap(cams, tr) for tr in TRUTHS},
        "fused": {tr: bootstrap(fused, tr) for tr in TRUTHS},
        "per_scene": {
            r["scene"]: {
                "cameras": len(r["cameras"]),
                "raw_ids": sum(u["raw_ids"] for u in r["cameras"]),
                "chains": sum(u["chains"] for u in r["cameras"]),
                "per_camera": {
                    a: {tr: pooled(r["cameras"], a, tr) for tr in TRUTHS} for a in ARMS
                },  # fmt: skip
                "fused": {a: {tr: pooled([r["fused"]], a, tr) for tr in TRUTHS} for a in ARMS},
                "seconds": r["seconds"],
            }
            for r in results
        },
        "live": {
            "per_camera": live_rate([e for r in results for e in r["live_cameras"]])
            | {"ci": live_bootstrap(results)},
            "fused": {
                a: sum(r["live_fused"][a]["violations"] for r in results)
                / max(1, sum(r["live_fused"][a]["shown"] for r in results))
                for a in ARMS
            }
            | {"shown": {a: sum(r["live_fused"][a]["shown"] for r in results) for a in ARMS}},
            "note": "the product counts raw ids live; chains are a whole-slot diagnostic",
        },
        "cameras": cams,
        "fused_scenes": fused,
        "seconds": round(time.time() - t0),
    }
    primary = TRUTHS[0]
    report["rule"] = {
        "per_camera (run)": rule(report["per_camera"][primary]),
        "fused (run --site)": rule(report["fused"][primary]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for surface in ("per_camera", "fused"):
        b = report[surface][primary]
        v, h = b["violation_rate"], b["hidden_share"]
        print(
            f"{surface}: violation rate ids {v['ids']['value']:.4f} chains "
            f"{v['chains']['value']:.4f} delta {v['delta']['value']:+.4f} "
            f"[{v['delta']['lo']:+.4f}, {v['delta']['hi']:+.4f}]; hidden share ids "
            f"{h['ids']['value']:.4f} chains {h['chains']['value']:.4f} delta "
            f"{h['delta']['value']:+.4f} [{h['delta']['lo']:+.4f}, {h['delta']['hi']:+.4f}]; "
            f"shown {b['shown']}"
        )
    print(json.dumps(report["rule"], indent=2))
    print(f"live per camera {report['live']['per_camera']['ci']}; fused {report['live']['fused']}")
    print(f"online chains equal batch stitch: {report['online_equals_batch']}; {OUT}")


if __name__ == "__main__":
    main()
