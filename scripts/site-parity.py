"""Map parity of the site recorder (improvements plan, phase 5, P1), as
pre-registered in docs/experiment-log.md, "Improvements plan, phase 5: multi-camera recording".

    uv run --no-sync python scripts/site-parity.py --site configs/site-071.yaml \
        --db data/site-071.db --out reports/site-parity-071.json

Recorder side: the site's fused dwell slots read back through read_slots
(k=5 on each slot), summed over every slot. Eval side: the fused export of the
same recipe (the site's recipe at its stride, replayed from the stride-2
TensorRT detection cache, no stitching) with the site's fusion, k=5 applied as
eval --cameras exports it (fused_export). Pass: TVD <= .01 between the two
maps normalised to one, over every cell of the grid.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from footfall_heatmaps.cli import data_dir, fused_export, track_source
from footfall_heatmaps.evaluation.application.evaluate_fusion import evaluate_fusion
from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from footfall_heatmaps.occupancy.application.read_slots import read_slots
from footfall_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE
from footfall_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
from footfall_heatmaps.site_recording import load_site

CACHED_STRIDE = 2
TVD_MAX = 0.01
END_OF_TIME = 4_102_444_800


def tvd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def recorded(site, db):
    store = SqliteSlotStore(db)
    try:
        slots = read_slots(store, site.id, 0, END_OF_TIME, K_MIN_PEOPLE)
    finally:
        store.close()
    dwell = [s for s in slots if s.layer == "dwell"]
    passage = [s for s in slots if s.layer == "passage"]
    total = sum(s.grid for s in dwell)
    shown = [s.grid[s.grid > 0] for s in passage]
    return total, {
        "slots": len(dwell),
        "incomplete": sum(s.incomplete for s in dwell),
        "n_unique": [s.n_unique for s in dwell],
        "min_people_in_a_shown_cell": float(min(g.min() for g in shown if g.size))
        if any(g.size for g in shown)
        else None,
        "cells_shown": int(sum(g.size for g in shown)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--scene", default="071")
    p.add_argument("--anchor", default="645")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    site = load_site(args.site)
    reader = NvidiaMtmcReader(data_dir())
    cameras = [c.split("/")[-1] for c in (cam.calibration for cam in site.cameras)]
    config = replace(site.config, stitch=None)
    run = (config, track_source(config, reader, detections_stride=CACHED_STRIDE))
    r = evaluate_fusion(
        reader,
        [(args.scene, args.anchor)],
        run,
        run,
        site.spec.cell_m,
        site.fusion,
        n_resamples=10,
        cameras=cameras,
    )
    export = fused_export(args.scene, cameras, r.fused[args.scene], site.fusion, "site recipe")
    ev = np.asarray(export["dwell"])
    rec, info = recorded(site, args.db)
    if ev.shape != rec.shape:
        raise SystemExit(f"grids differ: eval {ev.shape}, recorded {rec.shape}")
    value = tvd(rec, ev)
    report = {
        "site": args.site,
        "scene": args.scene,
        "cameras": cameras,
        "stride": config.stride,
        "eval_source": f"cached stride-{CACHED_STRIDE} TensorRT detections replayed at stride "
        f"{config.stride}, no stitching, whole-run weights",
        "fusion": vars(site.fusion),
        "k": K_MIN_PEOPLE,
        "tvd": round(value, 5),
        "tvd_max": TVD_MAX,
        "pass": value <= TVD_MAX,
        "dwell_total_s": {"recorded": float(rec.sum()), "eval": float(ev.sum())},
        "cells_above_zero": {"recorded": int((rec > 0).sum()), "eval": int((ev > 0).sum())},
        "recorded": info,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"P1 parity scene {args.scene}: TVD {value:.4f} (<= {TVD_MAX}: "
        f"{'pass' if report['pass'] else 'fail'}); dwell {rec.sum():.0f} vs {ev.sum():.0f} "
        f"person-s; {info['slots']} slots; {args.out}"
    )


if __name__ == "__main__":
    main()
