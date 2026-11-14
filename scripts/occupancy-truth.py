"""Truth of a slot's mean occupancy (improvements plan, phase 6, H4), as
pre-registered in docs/experiment-log.md: the ground-truth person-seconds inside the
coverage the slot shows (cells at least one of its cameras sees), during the
slot, over the observed slot seconds the recorder stored. Primary truth: the
people visible to the slot's cameras (a visible box in at least one of them,
eval's visible@ blocks; the research report's "person-seconds visible to camera
645"). Also reported: every person on that coverage (eval's full@ blocks),
which includes people hidden behind shelves that no camera can see, and the
visible truth on the shown cells only. Compared with the
mean occupancy the viewer shows for that slot (slot_reading on the slot read
back with k=5). Pass: within +-10 %.

    uv run --no-sync python scripts/occupancy-truth.py --site configs/site-071.yaml \
        --db data/site-071.db --out reports/occupancy-truth-071.json
    uv run --no-sync python scripts/occupancy-truth.py --config data/cam-645-s10.yaml \
        --db data/cam645.db --start 2026-11-06T09:00:00Z --out reports/occupancy-truth-645.json

Both truths are binned on the recording's grid by
evaluation.domain.bootstrap.block_layers, the dwell the evaluation scores
against; the coverage is Homography.sees per camera, as evaluate_fusion builds
it. A video's frame f is at start + f / fps.
"""

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from space_usage_heatmaps.evaluation.domain.bootstrap import block_layers
from space_usage_heatmaps.occupancy.domain.slots import SLOT_S, slot_reading

TOLERANCE = 0.10
END_OF_TIME = 4_102_444_800


def on_page(x):
    """The value as the page prints it: the API's 3 decimals, then one decimal,
    halves away from zero (Intl.NumberFormat)."""
    return float(Decimal(str(round(x, 3))).quantize(Decimal("0.1"), ROUND_HALF_UP))


def truth_person_seconds(spec, frames, world, fps, coverage, video_start, slot_start):
    """Truth person-seconds on the covered cells during [slot_start, slot_start + SLOT_S);
    frames: truth frame of each row, world: (N, 2) floor positions in metres."""
    frames = np.asarray(frames)
    t = video_start + frames / fps
    keep = (t >= slot_start) & (t < slot_start + SLOT_S)
    if not keep.any():
        return 0.0
    ids = np.zeros(int(keep.sum()), int)
    layers = block_layers(spec, ids, frames[keep], *np.asarray(world)[keep].T, fps,
                          int(frames.max()) + 1, 1)  # fmt: skip
    return float(layers.dwell.sum(0)[coverage].sum())


def main():
    from space_usage_heatmaps.cli import (
        calibration_of,
        data_dir,
        load_run_settings,
        utc_epoch,
    )
    from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import (
        NvidiaMtmcReader,
    )
    from space_usage_heatmaps.floorplan.domain.homography import Homography
    from space_usage_heatmaps.occupancy.application.read_slots import read_minutes, read_slots
    from space_usage_heatmaps.occupancy.domain.grid import GridSpec
    from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
    from space_usage_heatmaps.site_recording import load_site

    p = argparse.ArgumentParser()
    p.add_argument("--site", help="site config (its cameras, spec and start_utc)")
    p.add_argument("--config", help="one camera's run config (its calibration and extent)")
    p.add_argument("--start", help="with --config: the UTC wall time of the video's frame 0")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if bool(args.site) == bool(args.config):
        p.error("give one of --site and --config")
    if args.site:
        site = load_site(args.site)
        key, spec, start = site.id, site.spec, utc_epoch(site.start_utc)
        calibrations = [c.calibration for c in site.cameras]
    else:
        s = load_run_settings(args.config)
        key, start = s.camera_id, utc_epoch(args.start or s.start_utc)
        spec = GridSpec.covering(*s.extent_m, cell_m=s.cell_m)
        calibrations = [s.calibration]
    scene = calibrations[0].split("/")[1]
    cams = [calibration_of(c) for c in calibrations]
    coverage = np.zeros((spec.ny, spec.nx), bool)
    for cam in cams:
        coverage |= Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
    (fps,) = {cam.fps for cam in cams}
    reader = NvidiaMtmcReader(data_dir())
    truth = reader.floor_truth(scene, calibrations[0].split("/")[2])
    visible = np.zeros(len(truth.track_ids), bool)
    for c in calibrations:
        visible |= reader.floor_truth(scene, c.split("/")[2]).visible

    store = SqliteSlotStore(args.db)
    try:
        slots = read_slots(store, key, 0, END_OF_TIME)
        minutes = {m.start_utc: m for m in read_minutes(store, key, 0, END_OF_TIME)}
    finally:
        store.close()
    rows = []
    for s in (s for s in slots if s.layer == "dwell"):
        shown = slot_reading(s.grid, minutes.get(s.start_utc), s.incomplete)

        def person_s(rows, cells, start_utc=s.start_utc):
            return truth_person_seconds(
                spec, truth.frames[rows], truth.world[rows], fps, cells, start, start_utc
            )

        truth_s = person_s(visible, coverage)
        full_s = person_s(np.ones(len(visible), bool), coverage)
        shown_cells = person_s(visible, coverage & (s.grid > 0))
        observed = shown["observed_s"]
        mean_truth = truth_s / observed
        error = on_page(shown["mean_occupancy"]) / mean_truth - 1
        rows.append({
            "slot_start_utc": s.start_utc,
            "incomplete": s.incomplete,
            "observed_s": observed,
            "shown_mean_occupancy": shown["mean_occupancy"],
            "shown_mean_occupancy_on_page": on_page(shown["mean_occupancy"]),
            "shown_peak_occupancy": shown["peak_occupancy"],
            "shown_dwell_person_s": float(s.grid.sum()),
            "truth": "visible to at least one of the slot's cameras, on their coverage",
            "truth_person_s_in_coverage": truth_s,
            "truth_mean_occupancy": mean_truth,
            "relative_error": error,  # of the value on the page
            "relative_error_exact": shown["mean_occupancy"] / mean_truth - 1,
            "pass": abs(error) <= TOLERANCE,
            "truth_mean_occupancy_on_shown_cells_only": shown_cells / observed,
            "full_truth_mean_occupancy_in_coverage": full_s / observed,
            "relative_error_vs_full_truth": shown["mean_occupancy"] / (full_s / observed) - 1,
        })  # fmt: skip
    report = {
        "source": args.site or args.config,
        "db": args.db,
        "slot_key": key,
        "scene": scene,
        "cameras": calibrations,
        "coverage_cells": int(coverage.sum()),
        "fps": fps,
        "tolerance": TOLERANCE,
        "slots": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        print(
            f"{key} slot {r['slot_start_utc']}: shown {r['shown_mean_occupancy']:.3f}, truth "
            f"{r['truth_mean_occupancy']:.3f} people in view ({r['relative_error']:+.1%}; "
            f"{'pass' if r['pass'] else 'fail'})"
        )


if __name__ == "__main__":
    main()
