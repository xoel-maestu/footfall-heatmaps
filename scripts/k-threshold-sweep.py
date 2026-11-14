"""A higher id threshold per 0.5 m cell (improvements plan, phase 7, H8), as
pre-registered in docs/experiment-log.md, "Improvements plan, phase 7" (H8).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 uv run --no-sync python scripts/k-threshold-sweep.py
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 uv run --no-sync \
        python scripts/k-threshold-sweep.py --sealed T

Hypothesis: track ids overcount people (1.54 raw ids per true person in the
shown 0.5 m cells, 2.58 in the violating ones; H6 diagnostic), so requiring
more distinct ids per cell (T > 5) compensates for it.

Variants (raw track ids, no stitching, display and count on the 0.5 m cell):
T in {5 (today, H6's Z0), 7, 10, 15} distinct raw ids per cell and slot. The
same T applies live.

Recipe and slot: as scripts/k-zones-sweep.py (H6), whose helpers this script
imports. configs/e2a-weights.yaml at stride 10, replayed from the stride-2
detection cache, no stitch in the tracks, W1 fusion. Tune 071-075, every
calibrated camera. One slot per camera: the tracked rows in the truth frames
[first, last] of the scene. CPU only.

Choices fixed here before the first run (not in the pre-registration):
- Shown cell: distinct raw ids in the cell >= T (fused: the W1-weighted mean
  of the cameras' distinct ids, as run --site's fused passage, >= T) and its
  predicted raw dwell > 0. T=5 is H6's Z0 exactly.
- Violation: a shown cell whose true distinct people (primary: visible, as
  H3/H6; secondary: full) in the slot are < 5. The bar stays 5 people for
  every T: T counts ids, the promise counts people. Rate: violating over shown
  cells, pooled sum over sum over the bootstrap units (cameras; fused: scenes).
- Hidden mass (guard): true dwell (primary truth) in the cells with predicted
  dwell > 0 that the threshold hides, on the coverage (per camera: its own;
  fused: the joint coverage), over the true dwell on the coverage, pooled sum
  over sum. Guard: delta against T=5 (point) <= +.10, on both surfaces.
- TVD guard: dwell TVD at full@joint of the fused shown map (0.5 m grid,
  compare_layer's TVD formula), per scene, mean over the 5 scenes. Guard:
  delta against T=5 (point) <= +.02. Per camera TVD at full@coverage is
  reported, no rule.
- Target: pooled violation rate (point) <= .05 per camera and fused.
- CI: bootstrap over units (cameras; fused and TVD: scenes), 1000 resamples,
  seed 0 (numpy default_rng), percentile 95 %, every T paired with T=5 on the
  same resample.
- Selection: the smallest T that meets the target on both surfaces and holds
  both guards. If none, T stays 5 (no fallback variant).
- Live layer (reported for every T, no rule): instants every 60 s from first
  + 120 s to last; per camera a cell is shown where it had >= T distinct ids
  in the last 120 s (it then has an id seen in the window); fused: the
  weighted count >= T. Truth: the visible people in the cell in that window
  (fused: visible in any camera of the scene). Per camera CI over cameras.
- Sealed: `--sealed T` scores the selected T once on 076-080, with T=5
  alongside as the reference of its deltas (no selection uses it). Before any
  sealed data is read it creates reports/k-threshold-sealed.marker
  exclusively (O_EXCL) with T and the time; if the marker exists, it refuses
  to run. The marker file is kept with the evidence.
"""

import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "k_zones_sweep", Path(__file__).with_name("k-zones-sweep.py")
)
kz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kz)

from space_usage_heatmaps.cli import data_dir, load_fusion, source_sha256, track_source  # noqa: E402
from space_usage_heatmaps.evaluation.application.evaluate_oracle import recipe_floor_points  # noqa: E402
from space_usage_heatmaps.evaluation.domain.splits import SPLITS  # noqa: E402
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader  # noqa: E402
from space_usage_heatmaps.floorplan.domain.homography import Homography  # noqa: E402
from space_usage_heatmaps.occupancy.domain.fusion import CameraGrid, fusion_weights  # noqa: E402
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec  # noqa: E402
from space_usage_heatmaps.occupancy.domain.live import RECENT_S  # noqa: E402

RECIPE, STRIDE, CACHED_STRIDE, CELL_M = kz.RECIPE, kz.STRIDE, kz.CACHED_STRIDE, kz.CELL_M
PEOPLE = 5  # the promise: a shown cell had at least 5 people
THRESHOLDS = (5, 7, 10, 15)
BASE = "T5"
RESAMPLES, SEED = 1000, 0
TARGET, HIDDEN_GUARD, TVD_GUARD = 0.05, 0.10, 0.02
LIVE_STEP_S = 60
TRUTHS = ("visible", "full")
MARKER = Path("reports/k-threshold-sealed.marker")


def judge(shown, people):
    return {"shown": int(shown.sum()), "violations": int((shown & (people < PEOPLE)).sum())}


def measure_scene(args):
    scene, thresholds = args
    arms = {f"T{t}": t for t in thresholds}
    t0 = time.time()
    reader = NvidiaMtmcReader(data_dir())
    config = kz.run_config()
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
        rates.add(cam.fps)
        t = source.tracks(scene, c)
        ids_all, xs_all, ys_all, _ = recipe_floor_points(cam, t, config)
        keep = (t.frames >= first) & (t.frames <= last)
        order = np.argsort(t.frames[keep], kind="stable")
        ids = np.asarray(ids_all)[keep][order]
        frames = np.asarray(t.frames)[keep][order]
        xs, ys = xs_all[keep][order], ys_all[keep][order]
        coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
        truth = reader.floor_truth(scene, c)
        # Rows are masked per camera against truth_all, so they must align row for row.
        assert np.array_equal(truth.frames, truth_all.frames) and np.array_equal(
            truth.track_ids, truth_all.track_ids
        ), f"truth rows of camera {c} do not align with the scene's"
        seen_by_any |= truth.visible
        v = truth.visible
        samples = {
            "visible": (truth.track_ids[v], *truth.world[v].T),
            "full": (truth.track_ids, *truth.world.T),
        }
        people = {tr: kz.distinct(spec, *s, 1) for tr, s in samples.items()}
        tdwell = {tr: Grid.from_samples(spec, *s, cam.fps).dwell for tr, s in samples.items()}
        raw = Grid.from_samples(spec, ids, xs, ys, cam.fps)
        count = kz.distinct(spec, ids, xs, ys, 1)
        unit = {"scene": scene, "camera": c, "rows": int(len(ids)), "tvd_coverage": {}}
        for arm, k in arms.items():
            passed = count >= k
            shown = passed & (raw.dwell > 0)
            hidden = (raw.dwell > 0) & ~passed & coverage
            unit[arm] = {
                tr: judge(shown, people[tr])
                | {
                    "hidden_mass": float(tdwell[tr][hidden].sum()),
                    "mass": float(tdwell[tr][coverage].sum()),
                }
                for tr in TRUTHS
            }
            shown_map = np.where(passed, raw.dwell, 0.0)
            unit["tvd_coverage"][arm] = kz.tvd(shown_map, tdwell["full"], coverage, 1)
        per_camera.append(unit)
        cams.append((ids, frames, xs, ys, raw, coverage, truth, count, c))
    if len(rates) != 1:
        raise ValueError(f"scene {scene}: cameras at different frame rates {sorted(rates)}")
    (fps,) = rates

    # Fused slot: the threshold after fusion, as run --site.
    grids = [CameraGrid(c[8], c[4], c[5]) for c in cams]
    w = kz.unit_weights(fusion_weights(grids, fusion), 1)  # as H6's Z0
    joint = np.logical_or.reduce([g.coverage for g in grids])
    va = seen_by_any
    samples = {
        "visible": (truth_all.track_ids[va], *truth_all.world[va].T),
        "full": (truth_all.track_ids, *truth_all.world.T),
    }
    people = {tr: kz.distinct(spec, *s, 1) for tr, s in samples.items()}
    tdwell = {tr: Grid.from_samples(spec, *s, fps).dwell for tr, s in samples.items()}
    fcount = sum(w[i] * c[7] for i, c in enumerate(cams))
    fdwell = sum(w[i] * c[4].dwell for i, c in enumerate(cams))
    fused = {"scene": scene, "cameras": len(cams), "tvd_joint": {}}
    for arm, k in arms.items():
        passed = fcount >= k
        shown = passed & (fdwell > 0)
        hidden = (fdwell > 0) & ~passed & joint
        fused[arm] = {
            tr: judge(shown, people[tr])
            | {
                "hidden_mass": float(tdwell[tr][hidden].sum()),
                "mass": float(tdwell[tr][joint].sum()),
            }
            for tr in TRUTHS
        }
        fused["tvd_joint"][arm] = kz.tvd(np.where(passed, fdwell, 0.0), tdwell["full"], joint, 1)

    # Live layer: 120 s windows every 60 s.
    window = round(RECENT_S * fps)
    instants = list(range(first + window, last + 1, round(LIVE_STEP_S * fps)))
    t_ids, t_frames, t_world = truth_all.track_ids, truth_all.frames, truth_all.world
    live_cam = [{a: {"t": {"shown": 0, "violations": 0}} for a in arms} for _ in cams]
    live_fused = {a: {"shown": 0, "violations": 0} for a in arms}
    for f0 in instants:
        in_t = (t_frames >= f0 - window) & (t_frames <= f0)
        any_rows = in_t & seen_by_any
        counts = []
        for i, (ids, frames, xs, ys, _, _, truth, _, _) in enumerate(cams):
            rows = (frames >= f0 - window) & (frames <= f0)
            counts.append(kz.distinct(spec, ids[rows], xs[rows], ys[rows], 1))
            tv = in_t & truth.visible
            tp = kz.distinct(spec, t_ids[tv], *t_world[tv].T, 1)
            for arm, k in arms.items():
                e = judge(counts[i] >= k, tp)
                for key in e:
                    live_cam[i][arm]["t"][key] += e[key]
        fc = sum(w[i] * counts[i] for i in range(len(cams)))
        tp = kz.distinct(spec, t_ids[any_rows], *t_world[any_rows].T, 1)
        for arm, k in arms.items():
            e = judge(fc >= k, tp)
            for key in e:
                live_fused[arm][key] += e[key]
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


def bootstrap(units, truth, arms):
    """Pooled violation rate and hidden share per arm, paired deltas against T=5."""
    rng = np.random.default_rng(SEED)
    n = len(units)
    others = [a for a in arms if a != BASE]

    def stats(sample):
        rate = {a: pooled(sample, a, truth) for a in arms}
        hidden = {a: pooled(sample, a, truth, "hidden_mass", "mass") for a in arms}
        return {
            "violation_rate": rate | {f"delta_{a}": rate[a] - rate[BASE] for a in others},
            "hidden_share": hidden | {f"delta_{a}": hidden[a] - hidden[BASE] for a in others},
        }

    point = stats(units)
    draws = [stats([units[i] for i in rng.integers(0, n, n)]) for _ in range(RESAMPLES)]
    out = {
        m: {k: kz.with_ci(v, [x[m][k] for x in draws]) for k, v in point[m].items()} for m in point
    }
    out["shown"] = {a: sum(u[a][truth]["shown"] for u in units) for a in arms}
    out["violations"] = {a: sum(u[a][truth]["violations"] for u in units) for a in arms}
    return out


def tvd_bootstrap(values, arms):
    rng = np.random.default_rng(SEED)
    n = len(values)

    def stats(sample):
        m = {a: float(np.mean([s[a] for s in sample])) for a in arms}
        return m | {f"delta_{a}": m[a] - m[BASE] for a in arms if a != BASE}

    point = stats(values)
    draws = [stats([values[i] for i in rng.integers(0, n, n)]) for _ in range(RESAMPLES)]
    return {k: kz.with_ci(v, [x[k] for x in draws]) for k, v in point.items()}


def live_bootstrap(units, arms):
    rng = np.random.default_rng(SEED)
    n = len(units)
    out = {}
    for a in arms:
        draws = [
            pooled([units[i] for i in rng.integers(0, n, n)], a, "t") for _ in range(RESAMPLES)
        ]
        out[a] = kz.with_ci(pooled(units, a, "t"), draws) | {
            "shown": sum(u[a]["t"]["shown"] for u in units),
            "violations": sum(u[a]["t"]["violations"] for u in units),
        }
    return out


def decide(report, arms):
    p = TRUTHS[0]
    cam, fus, tvd_j = report["per_camera"][p], report["fused"][p], report["tvd_full_joint"]
    rules = {}
    for a in arms:

        def delta(s, m, a=a):
            return 0.0 if a == BASE else s[m][f"delta_{a}"]["value"]

        checks = {
            "per_camera_le_target": cam["violation_rate"][a]["value"] <= TARGET,
            "fused_le_target": fus["violation_rate"][a]["value"] <= TARGET,
            "hidden_guard_per_camera": delta(cam, "hidden_share") <= HIDDEN_GUARD,
            "hidden_guard_fused": delta(fus, "hidden_share") <= HIDDEN_GUARD,
            "tvd_guard": (0.0 if a == BASE else tvd_j[f"delta_{a}"]["value"]) <= TVD_GUARD,
        }
        checks["passes"] = all(checks.values())
        rules[a] = checks
    passing = [a for a in arms if rules[a]["passes"]]
    if passing:
        return {"thresholds": rules, "chosen": passing[0], "why": "smallest T that passes"}
    return {"thresholds": rules, "chosen": BASE, "why": "no T passes; T stays 5"}


def run(split, thresholds):
    t0 = time.time()
    arms = [f"T{t}" for t in thresholds]
    scenes = [s for s, _ in SPLITS[split]]
    with ProcessPoolExecutor(int(os.environ.get("SWEEP_WORKERS", 3))) as pool:
        results = list(pool.map(measure_scene, [(s, thresholds) for s in scenes]))
    cams = [u for r in results for u in r["cameras"]]
    fused = [r["fused"] for r in results]
    live_units = [u for r in results for u in r["live_cameras"]]
    report = {
        "hypothesis": "H8: distinct raw ids per 0.5 m cell >= T, T in "
        f"{thresholds}; violation: shown cell with < {PEOPLE} true people",
        "recipe": f"{RECIPE} at stride {STRIDE} from the stride-{CACHED_STRIDE} detection cache, "
        "no stitch in the tracks, W1 fusion, raw ids",
        "thresholds": list(thresholds),
        "split": split,
        "scenes": scenes,
        "n_resamples": RESAMPLES,
        "seed": SEED,
        "interval": "bootstrap over units (cameras; fused, TVD: scenes), percentile 95 %, "
        "paired with T=5",
        "primary_truth": TRUTHS[0],
        "source_sha256": source_sha256(),
        "per_camera": {tr: bootstrap(cams, tr, arms) for tr in TRUTHS},
        "fused": {tr: bootstrap(fused, tr, arms) for tr in TRUTHS},
        "tvd_full_joint": tvd_bootstrap([f["tvd_joint"] for f in fused], arms),
        "tvd_full_coverage_per_camera": tvd_bootstrap([u["tvd_coverage"] for u in cams], arms),
        "per_scene": {
            r["scene"]: {
                "cameras": len(r["cameras"]),
                "per_camera": {a: pooled(r["cameras"], a, "visible") for a in arms},
                "fused": {a: pooled([r["fused"]], a, "visible") for a in arms},
                "seconds": r["seconds"],
            }
            for r in results
        },
        "live": {
            "per_camera": live_bootstrap(live_units, arms),
            "fused": {
                a: {
                    "value": sum(r["live_fused"][a]["violations"] for r in results)
                    / max(1, sum(r["live_fused"][a]["shown"] for r in results)),
                    "shown": sum(r["live_fused"][a]["shown"] for r in results),
                }
                for a in arms
            },
        },
        "cameras": cams,
        "fused_scenes": fused,
    }
    if split == "tune":
        report["rule"] = decide(report, arms)
    report["seconds"] = round(time.time() - t0)
    p = TRUTHS[0]
    for surface in ("per_camera", "fused"):
        b = report[surface][p]
        for a in arms:
            v, h = b["violation_rate"][a], b["hidden_share"][a]
            line = f"{surface} {a}: rate {v['value']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]"
            line += f" shown {b['shown'][a]}; hidden {h['value']:.4f}"
            if a != BASE:
                dv, dh = b["violation_rate"][f"delta_{a}"], b["hidden_share"][f"delta_{a}"]
                line += f"; delta rate {dv['value']:+.4f} [{dv['lo']:+.4f}, {dv['hi']:+.4f}]"
                line += f", delta hidden {dh['value']:+.4f} [{dh['lo']:+.4f}, {dh['hi']:+.4f}]"
            print(line)
    for key in ("tvd_full_joint", "tvd_full_coverage_per_camera"):
        print(key, {k: round(v["value"], 4) for k, v in report[key].items()})
    print(
        "live per camera",
        {a: round(v["value"], 4) for a, v in report["live"]["per_camera"].items()},
    )
    print("live fused", {a: round(v["value"], 4) for a, v in report["live"]["fused"].items()})
    if "rule" in report:
        print(json.dumps(report["rule"], indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", type=int, help="score this T once on the sealed split")
    args = ap.parse_args()
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    if args.sealed is None:
        report, out = run("tune", THRESHOLDS), Path("reports/k-threshold-tune.json")
    else:
        if args.sealed not in THRESHOLDS or args.sealed == 5:
            raise SystemExit(f"--sealed takes a T of {THRESHOLDS[1:]}")
        try:
            fd = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"{MARKER} exists: the sealed split was already scored") from None
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"T": args.sealed, "started": time.time()}) + "\n")
        report = run("sealed", (5, args.sealed))
        out = Path("reports/k-threshold-sealed.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{out}, {report['seconds']} s")


if __name__ == "__main__":
    main()
