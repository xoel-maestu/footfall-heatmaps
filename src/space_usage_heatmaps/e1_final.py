"""space-usage eval --suite e1-final: the E1 report of the accepted config.

It gathers, for configs/h5-best.yaml, the tune and sealed map metrics with the
20 % rule and the spec 7 targets (recalibrated once after iteration 0), the
sealed confirmation of every accepted step, the external datasets, the
label-free metrics with their correlation against the true error, and the
bench's fps and VRAM. Tracks come from the caches (the detector does not run
where a valid cache exists); an eval report under reports/ is reused when it
was made by the same code, config and resamples. Writes reports/e1-final.json
and reports/e1-final.md.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from space_usage_heatmaps import cli
from space_usage_heatmaps.evaluation.application.label_free_validation import (
    LABEL_FREE,
    TRUE_ERRORS,
    validate_label_free,
)
from space_usage_heatmaps.evaluation.domain.splits import SPLITS

SUITE_CONFIG = "configs/h5-best.yaml"
# Each accepted step against the config it was accepted over (iterations.md).
# "accuracy": accepted on a TVD gain, holds when the sealed CI is below 0.
# "no-loss": an engine change, holds when the sealed CI does not sit above 0.
CONFIRMATIONS = (
    ("H4 auto-track foot point", "configs/h4-auto-track.yaml", "configs/iter0.yaml", "accuracy"),
    ("H1 YOLOX-s detector", "configs/h1-yolox-s.yaml", "configs/h4-auto-track.yaml", "accuracy"),
    ("H2a input 960", "configs/h1-best.yaml", "configs/h1-yolox-s.yaml", "accuracy"),
    ("H5 TensorRT fp16", "configs/h5-best.yaml", "configs/h1-best.yaml", "no-loss"),
)
TRUTH = "visible@coverage"  # E1 primary block, dwell layer
# Spec 7 after the one recalibration (iterations.md, "Target recalibration").
MAP_TARGETS = {"tvd": 0.30, "mass_error": 0.10, "wape_zone": 0.30, "emd_m": 1.2}
SEALED_MAX_DEGRADATION = 0.20
FPS_MIN, VRAM_MAX_MB = 15.0, 3500.0
DRIFT = (
    "camera drift not computed: it compares each frame's edges with a background "
    "reference, and the evaluation keeps positions only, never frames; the NVIDIA "
    "cameras are static, so there is no drift to validate it against"
)
JSON_PATH, MD_PATH = Path("reports/e1-final.json"), Path("reports/e1-final.md")


def _fresh(path, config_paths, resamples):
    """The stored eval report at path when the same code, configs and resamples made it."""
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    configs = {cli.config_name(p): vars(cli.load_config(p)) for p in config_paths}
    if (
        report.get("source_sha256") != cli.source_sha256()
        or report.get("n_resamples") != resamples
        or report.get("configs") != json.loads(json.dumps(configs))
    ):
        return None
    return report


def config_eval(config, compare, split, cell, resamples, log):
    paths = [config] + ([compare] if compare else [])
    names = [cli.config_name(p) for p in paths]
    path = cli.eval_report_path(names, split, cell)
    if (report := _fresh(path, paths, resamples)) is not None:
        log(f"{path}: reused")
        return report
    log(f"{path}: scoring {' vs '.join(names)} on {split}")
    report, _, _ = cli.config_eval(config, compare, split, cell, resamples)
    cli.write_report(path, report)
    return report


def gather(cell, resamples, log):
    """Every input of the E1 report, from caches and stored reports."""
    from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import (
        NvidiaMtmcReader,
    )

    evals = {
        split: config_eval(SUITE_CONFIG, None, split, cell, resamples, log) for split in SPLITS
    }
    confirmations = [
        {
            "label": label,
            "config": cli.config_name(config),
            "baseline": cli.config_name(baseline),
            "kind": kind,
            **{
                split: config_eval(config, baseline, split, cell, resamples, log)
                for split in ("sealed", "tune")
            },
        }
        for label, config, baseline, kind in CONFIRMATIONS
    ]
    external = {}
    for dataset in cli.EXTERNAL:
        log(f"external {dataset}")
        external[dataset] = cli.dataset_eval(dataset, SUITE_CONFIG)
    log("label-free metrics on tune and sealed")
    reader = NvidiaMtmcReader(cli.data_dir())
    config = cli.load_config(SUITE_CONFIG)
    scenes = [s for split in ("tune", "sealed") for s in SPLITS[split]]
    validation = validate_label_free(reader, scenes, config, cli.track_source(config, reader), cell)
    try:
        bench = json.loads(cli.bench_report_path(SUITE_CONFIG).read_text())
    except (OSError, ValueError):
        bench = None
    return {
        "evals": evals,
        "confirmations": confirmations,
        "external": external,
        "label_free": validation,
        "bench": bench,
    }


def _pooled(report, name, metric):
    return report["results"][TRUTH]["pooled"][name]["dwell"][metric]


def _degradation(tune, sealed, higher_is_better=False):
    if tune is None or sealed is None or tune == 0:
        return None
    return (tune - sealed) / tune if higher_is_better else (sealed - tune) / tune


def assemble(inputs):
    """The E1 summary (reports/e1-final.json) from the gathered inputs."""
    name = cli.config_name(SUITE_CONFIG)
    tune, sealed = inputs["evals"]["tune"], inputs["evals"]["sealed"]
    metrics = {}
    for metric, target in MAP_TARGETS.items():
        t, s = _pooled(tune, name, metric), _pooled(sealed, name, metric)
        d = _degradation(t["value"], s["value"])
        metrics[metric] = {
            "tune": t,
            "sealed": s,
            "degradation": d,
            "within_20pct": d is not None and d <= SEALED_MAX_DEGRADATION,
            "target": target,
            "tune_meets_target": t["value"] is not None and t["value"] <= target,
            "sealed_meets_target": s["value"] is not None and s["value"] <= target,
        }
    tracking = {}
    for metric in ("hota", "idf1"):
        t = tune["tracking"]["pooled"][name][metric]
        s = sealed["tracking"]["pooled"][name][metric]
        d = _degradation(t, s, higher_is_better=True)
        tracking[metric] = {
            "tune": t,
            "sealed": s,
            "degradation": d,
            "within_20pct": d is not None and d <= SEALED_MAX_DEGRADATION,
        }
    scenes = {
        split: {
            key: v[name]["dwell"]["tvd"]["value"]
            for key, v in r["results"][TRUTH]["scenes"].items()
        }
        for split, r in inputs["evals"].items()
    }
    full = {
        split: r["results"]["full@coverage"]["pooled"][name]["dwell"]["tvd"]
        for split, r in inputs["evals"].items()
    }
    confirmations = []
    for c in inputs["confirmations"]:
        delta = {split: c[split]["results"][TRUTH]["delta"]["pooled"]["dwell"]["tvd"]
                 for split in ("tune", "sealed")}  # fmt: skip
        s = delta["sealed"]
        if c["kind"] == "accuracy":
            holds = s["hi"] is not None and s["hi"] < 0 and delta["tune"]["value"] < 0
        else:
            holds = s["lo"] is not None and s["lo"] <= 0
        confirmations.append(
            {k: c[k] for k in ("label", "config", "baseline", "kind")}
            | {"delta_tvd": delta, "holds": holds}
        )
    external = _external(inputs["external"], name)
    speed = _speed(inputs["bench"])
    label_free = _label_free(inputs["label_free"])
    wildtrack = external.get("wildtrack") or {}
    accepted = (
        all(m["within_20pct"] and m["sealed_meets_target"] for m in metrics.values())
        and bool(wildtrack.get("meets_target"))
        and bool(speed and speed["fps_meets"] and speed["vram_meets"])
    )
    return {
        "suite": "e1-final",
        "config": name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_sha256": cli.source_sha256(),
        "map": {
            "block": TRUTH,
            "layer": "dwell",
            "cell_m": tune["cell_m"],
            "n_resamples": tune["n_resamples"],
            "sealed_max_degradation": SEALED_MAX_DEGRADATION,
            "metrics": metrics,
            "tvd_per_scene": scenes,
            "full_coverage_tvd": full,
        },
        "tracking": tracking,
        "confirmations": confirmations,
        "external": external,
        "speed": speed,
        "label_free": label_free,
        "e1_accepted": accepted,
    }


def _external(reports, name):
    out = {}
    for dataset, r in reports.items():
        if r is None:
            out[dataset] = None
            continue
        t = r["tracking"]
        entry = {
            "scenes": len(r["scenes"]),
            "hota": t["pooled"][name]["hota"],
            "idf1": t["pooled"][name]["idf1"],
        }
        if dataset == "mot17":
            entry["per_sequence"] = {
                k.split("/")[0]: {"hota": v["hota"], "idf1": v["idf1"]}
                for k, v in t["scenes"][name].items()
            }
        if "ground_position" in r:
            g, p = r["ground_position"], r["ground_plane"]

            def view(v):
                keys = ("median_m", "p90_m", "recall", "precision", "matched")
                return {k: v[k] for k in keys}

            entry |= {
                "camera": g["camera"],
                "camera_height_m": g["camera_height_m"],
                "head_rule": g["head_rule"],
                "image_pairing": view(g),
                "floor_pairing": view(p) | {"gate_m": p["gate_m"]},
                "target_m": g["target_m"],
                "meets_target": g["meets_target"],
            }
        out[dataset] = entry
    return out


def _speed(bench):
    if bench is None:
        return None
    return {
        k: bench.get(k) for k in ("fps_mean", "fps_p5", "vram_peak_mb", "minutes", "measured_at")
    } | {
        "engine": bench.get("engine"),
        "fps_meets": bench["fps_mean"] >= FPS_MIN,
        "vram_meets": bench["vram_peak_mb"] <= VRAM_MAX_MB,
        "fps_min": FPS_MIN,
        "vram_max_mb": VRAM_MAX_MB,
    }


def _number(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _label_free(v):
    return {
        "scenes": [
            {
                "scene": s.scene,
                "camera": s.camera,
                "split": next(k for k, sc in SPLITS.items() if (s.scene, s.camera) in sc),
                **s.label_free.as_dict(),
                **{f"true_{k}": x for k, x in s.true_error.items()},
                "blocks": s.blocks,
            }
            for s in v.scenes
        ],
        "correlations": {
            level: {k: {m: _number(x) for m, x in c.items()} for k, c in pairs.items()}
            for level, pairs in v.correlations.items()
        },
        "true_error": f"{TRUTH} dwell, whole run per scene and per 30 s block",
        "drift": DRIFT,
    }


def _f(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci(e, digits=3):
    if e["lo"] is None:
        return _f(e["value"], digits)
    return f"{_f(e['value'], digits)} [{_f(e['lo'], digits)}, {_f(e['hi'], digits)}]"


def _pct(v):
    return "n/a" if v is None else f"{100 * v:+.0f} %"


def _yes(v):
    return "pass" if v else "fail"


TITLES = {
    "tvd": "TVD",
    "mass_error": "Mass error",
    "wape_zone": "Zone WAPE",
    "emd_m": "EMD (m)",
    "hota": "HOTA",
    "idf1": "IDF1",
}


def render_markdown(s):
    m = s["map"]
    lines = [
        "# E1 final report",
        "",
        f"Config `{s['config']}`. Dwell maps on {m['block']}, {m['cell_m']} m cells, pooled over "
        f"5 scenes per split, 95 % paired block-bootstrap CI ({m['n_resamples']} resamples). "
        "Targets are spec 7 after its one recalibration; the sealed split must stay within "
        f"{100 * m['sealed_max_degradation']:.0f} % of tune.",
        "",
        f"E1 accepted: **{'yes' if s['e1_accepted'] else 'no'}**.",
        "",
        "## Tune and sealed",
        "",
        "| Metric | Tune | Sealed | Degradation | Within 20 % | Target | Tune | Sealed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, x in m["metrics"].items():
        lines.append(
            f"| {TITLES[k]} | {_ci(x['tune'])} | {_ci(x['sealed'])} | {_pct(x['degradation'])} "
            f"| {_yes(x['within_20pct'])} | ≤ {x['target']:g} | {_yes(x['tune_meets_target'])} "
            f"| {_yes(x['sealed_meets_target'])} |"
        )
    for k, x in s["tracking"].items():
        lines.append(
            f"| {TITLES[k]} | {_f(x['tune'])} | {_f(x['sealed'])} | {_pct(x['degradation'])} "
            f"| {_yes(x['within_20pct'])} | guard only | | |"
        )
    full = m["full_coverage_tvd"]
    lines += [
        f"| TVD full@coverage (phase 9 target) | {_ci(full['tune'])} | {_ci(full['sealed'])} "
        f"| {_pct(_degradation(full['tune']['value'], full['sealed']['value']))} | | | | |",
        "",
        "TVD per scene: "
        + "; ".join(
            f"{split} " + ", ".join(f"{k.split('/')[0]} {_f(v)}" for k, v in sc.items())
            for split, sc in m["tvd_per_scene"].items()
        )
        + ".",
        "",
        "## Sealed confirmation of each accepted step",
        "",
        "Paired ΔTVD (config minus the one it was accepted over).",
        "",
        "| Step | Config vs baseline | ΔTVD tune | ΔTVD sealed | Holds |",
        "|---|---|---|---|---|",
    ]
    for c in s["confirmations"]:
        lines.append(
            f"| {c['label']} | `{c['config']}` vs `{c['baseline']}` "
            f"| {_ci(c['delta_tvd']['tune'])} | {_ci(c['delta_tvd']['sealed'])} "
            f"| {'yes' if c['holds'] else 'no'} ({c['kind']}) |"
        )
    lines += ["", "## External datasets (evaluation only, never tuned on)", ""]
    ext = s["external"]
    w = ext.get("wildtrack")
    if w:
        rule = "on" if w["head_rule"] else "off, camera below the heads"
        lines += [
            f"WILDTRACK camera {w['camera']}, {w['camera_height_m']:.2f} m high "
            f"(auto-track head rule {rule}). Target: median ≤ {100 * w['target_m']:.0f} cm.",
            "",
            "| Pairing | Median | p90 | Recall | Precision | Pairs |",
            "|---|---|---|---|---|---|",
        ]
        for label, v in (
            ("image, IoU ≥ 0.5", w["image_pairing"]),
            (f"floor, within {w['floor_pairing']['gate_m']:g} m", w["floor_pairing"]),
        ):
            lines.append(
                f"| {label} | {_cm(v['median_m'])} | {_cm(v['p90_m'])} | {_f(v['recall'], 2)} "
                f"| {_f(v['precision'], 2)} | {v['matched']} |"
            )
        lines += ["", f"WILDTRACK target: {_yes(w['meets_target'])}.", ""]
    lines += ["| Dataset | Sequences | HOTA | IDF1 |", "|---|---|---|---|"]
    for dataset, label in (("wildtrack", "WILDTRACK"), ("caviar", "CAVIAR"), ("mot17", "MOT17")):
        e = ext.get(dataset)
        if e is None:
            lines.append(f"| {label} | missing | | |")
            continue
        lines.append(f"| {label} | {e['scenes']} | {_f(e['hota'])} | {_f(e['idf1'])} |")
        for seq, v in (e.get("per_sequence") or {}).items():
            lines.append(f"| {seq} | 1 | {_f(v['hota'])} | {_f(v['idf1'])} |")
    sp = s["speed"]
    lines += ["", "## Speed", ""]
    if sp is None:
        lines.append(
            "No bench report: run `uv run space-usage bench --config configs/h5-best.yaml`."
        )
    else:
        lines += [
            f"`space-usage bench`, {sp['minutes']:g} min on one 1080p camera ({sp['engine']}): "
            f"{_f(sp['fps_mean'], 1)} fps mean, {_f(sp['fps_p5'], 1)} fps p5 "
            f"(target ≥ {sp['fps_min']:g}: {_yes(sp['fps_meets'])}); VRAM peak "
            f"{_f(sp['vram_peak_mb'], 0)} MB (target ≤ {sp['vram_max_mb']:g} MB: "
            f"{_yes(sp['vram_meets'])}). Measured {sp['measured_at']}.",
        ]
    lf = s["label_free"]
    lines += [
        "",
        "## Label-free metrics against the true error",
        "",
        f"True error: {lf['true_error']}.",
        "",
        "| Scene | Split | Conservation error | Interior births | Mean track length (s) "
        "| Tracks | True TVD | True mass error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in lf["scenes"]:
        lines.append(
            f"| {r['scene']}/{r['camera']} | {r['split']} | {_f(r['conservation_error'])} "
            f"| {_f(r['interior_birth_fraction'])} | {_f(r['mean_track_length_s'], 1)} "
            f"| {r['tracks']} | {_f(r['true_tvd'])} | {_f(r['true_mass_error'])} |"
        )
    lines += [
        "",
        "Correlation with the true error. Per scene, and per 30 s block pooled over every "
        "scene (blocks of one scene are not independent: the block n overstates the evidence).",
        "",
        "| Label-free metric | True error | Spearman, scenes (n) | Pearson, scenes "
        "| Spearman, blocks (n) | Pearson, blocks |",
        "|---|---|---|---|---|---|",
    ]
    c = lf["correlations"]
    for metric in LABEL_FREE:
        for error in TRUE_ERRORS:
            a, b = c["scene"][f"{metric}~{error}"], c["block"][f"{metric}~{error}"]
            lines.append(
                f"| {metric.replace('_', ' ')} | {error.replace('_', ' ')} "
                f"| {_f(a['spearman'], 2)} ({a['n']}) | {_f(a['pearson'], 2)} "
                f"| {_f(b['spearman'], 2)} ({b['n']}) | {_f(b['pearson'], 2)} |"
            )
    lines += ["", f"Not computed: {lf['drift']}.", ""]
    return "\n".join(lines)


def _cm(v):
    return "n/a" if v is None else f"{100 * v:.1f} cm"


def run_e1_final(args, log=lambda line: print(line, flush=True)):
    inputs = gather(args.cell, args.resamples, log)
    summary = assemble(inputs)
    cli.write_report(JSON_PATH, summary)
    MD_PATH.write_text(render_markdown(summary))
    print(render_markdown(summary))
    print(f"report {JSON_PATH}\nsummary {MD_PATH}")
    return 0
