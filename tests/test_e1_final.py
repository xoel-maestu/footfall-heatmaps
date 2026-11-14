"""footfall eval --suite e1-final: the E1 summary from gathered inputs."""

import json

import numpy as np

from footfall_heatmaps import cli, e1_final
from footfall_heatmaps.evaluation.application.label_free_validation import (
    LABEL_FREE,
    TRUE_ERRORS,
    LabelFreeValidation,
    SceneLabelFree,
)
from footfall_heatmaps.evaluation.domain.label_free import LabelFree

NAME = "configs_h5-best"


def est(value, lo=None, hi=None):
    return {"value": value, "lo": lo, "hi": hi}


def eval_report(split, tvd, mass, wape, emd, hota, idf1, delta=None, tracking_delta=(0.0, 0.0)):
    dwell = {"tvd": est(tvd), "mass_error": est(mass), "wape_zone": est(wape), "emd_m": est(emd)}
    results = {
        "visible@coverage": {
            "pooled": {NAME: {"dwell": dwell}},
            "scenes": {"071/645": {NAME: {"dwell": {"tvd": est(tvd)}}}},
        },
        "full@coverage": {"pooled": {NAME: {"dwell": {"tvd": est(tvd + 0.1)}}}},
    }
    tracking = {"pooled": {NAME: {"hota": hota, "idf1": idf1}}}
    if delta is not None:
        results["visible@coverage"]["delta"] = {"pooled": {"dwell": {"tvd": est(*delta)}}}
        d_hota, d_idf1 = tracking_delta
        tracking["delta"] = {"pooled": {"hota": d_hota, "deta": 0.0, "assa": 0.0, "idf1": d_idf1}}
    return {
        "split": split,
        "cell_m": 0.5,
        "n_resamples": 1000,
        "results": results,
        "tracking": tracking,
    }


def inputs():
    lf = LabelFree(0.5, 0.4, 9.0, 100)
    scene = SceneLabelFree("071", "645", lf, {"tvd": 0.3, "mass_error": 0.15}, 27)
    corr = {f"{m}~{e}": {"spearman": 0.5, "pearson": np.nan, "n": 10}
            for m in LABEL_FREE for e in TRUE_ERRORS}  # fmt: skip
    wild = {
        "scenes": ["wildtrack/C5"],
        "tracking": {"pooled": {NAME: {"hota": 0.27, "idf1": 0.31}}},
        "ground_position": {
            "camera": "C5", "camera_height_m": 1.68, "head_rule": False, "target_m": 0.3,
            "meets_target": True, "median_m": 0.218, "p90_m": 0.451, "recall": 0.71,
            "precision": 0.32, "matched": 2633,
        },
        "ground_plane": {"median_m": 0.215, "p90_m": 0.43, "recall": 0.76, "precision": 0.35,
                         "matched": 2828, "gate_m": 1.0},
    }  # fmt: skip
    mot = {
        "scenes": ["MOT17-02/1", "MOT17-04/1"],
        "tracking": {
            "pooled": {NAME: {"hota": 0.4, "idf1": 0.5}},
            "scenes": {NAME: {"MOT17-02/1": {"hota": 0.3, "idf1": 0.4},
                              "MOT17-04/1": {"hota": 0.5, "idf1": 0.6}}},
        },
    }  # fmt: skip
    return {
        "evals": {
            "tune": eval_report("tune", 0.308, 0.154, 0.231, 1.50, 0.237, 0.182),
            "sealed": eval_report("sealed", 0.419, 0.297, 0.380, 3.19, 0.218, 0.189),
        },
        "confirmations": [
            {"label": "H4", "config": "a", "baseline": "b", "kind": "accuracy",
             "tune": eval_report("tune", 0, 0, 0, 0, 0, 0, (-0.248, -0.26, -0.23)),
             "sealed": eval_report("sealed", 0, 0, 0, 0, 0, 0, (-0.191, -0.203, -0.18))},
            {"label": "H9", "config": "c", "baseline": "d", "kind": "accuracy",
             "tune": eval_report("tune", 0, 0, 0, 0, 0, 0, (-0.02, -0.03, -0.01)),
             "sealed": eval_report("sealed", 0, 0, 0, 0, 0, 0, (-0.01, -0.02, 0.001),
                                   (0.02, -0.012))},
            {"label": "H5", "config": "e", "baseline": "f", "kind": "no-loss",
             "tune": eval_report("tune", 0, 0, 0, 0, 0, 0, (-0.001, -0.002, 0.0)),
             "sealed": eval_report("sealed", 0, 0, 0, 0, 0, 0, (-0.001, -0.003, 0.0))},
        ],  # fmt: skip
        "external": {"wildtrack": wild, "caviar": None, "mot17": mot},
        "label_free": LabelFreeValidation([scene], {}, {"scene": corr, "block": corr}),
        "bench": {"fps_mean": 42.4, "fps_p5": 39.9, "vram_peak_mb": 174.0, "minutes": 30,
                  "measured_at": "2026-11-06T00:00:00+00:00", "engine": "onnx-trt"},
    }  # fmt: skip


def test_the_summary_marks_the_20_percent_rule_and_the_targets():
    s = e1_final.assemble(inputs())

    tvd = s["map"]["metrics"]["tvd"]
    assert round(tvd["degradation"], 2) == 0.36 and tvd["within_20pct"] is False
    assert tvd["tune_meets_target"] is False and tvd["sealed_meets_target"] is False
    wape = s["map"]["metrics"]["wape_zone"]
    assert wape["tune_meets_target"] is True and wape["sealed_meets_target"] is False
    assert [c["holds"] for c in s["confirmations"]] == [True, False, True]
    assert s["external"]["wildtrack"]["floor_pairing"]["gate_m"] == 1.0
    assert s["external"]["caviar"] is None
    assert s["speed"]["fps_meets"] and s["speed"]["vram_meets"]
    assert s["e1_accepted"] is False
    json.dumps(s, allow_nan=False)  # NaN correlations become null


def test_the_markdown_has_every_section_and_says_what_is_missing():
    md = e1_final.render_markdown(e1_final.assemble(inputs()))

    assert "E1 accepted: **no**" in md
    assert "| TVD | 0.308 | 0.419 | +36 % | fail | ≤ 0.3 | fail | fail |" in md
    assert "| H9 |" in md and "| no (accuracy) |" in md
    assert "| floor, within 1 m | 21.5 cm | 43.0 cm | 0.76 | 0.35 | 2828 |" in md
    assert "| CAVIAR | missing | | |" in md and "| MOT17-04 | 1 | 0.500 | 0.600 |" in md
    assert "42.4 fps mean" in md and "Not computed: camera drift, it compares" in md
    assert "camera drift not computed" not in md
    assert "| conservation error | tvd | 0.50 (10) | n/a |" in md


def test_eval_suite_writes_the_json_and_the_markdown(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(e1_final, "gather", lambda cell, resamples, log: inputs())

    assert cli.main(["eval", "--suite", "e1-final"]) == 0

    s = json.loads((tmp_path / "reports" / "e1-final.json").read_text())
    assert s["suite"] == "e1-final" and s["config"] == NAME
    assert (tmp_path / "reports" / "e1-final.md").read_text().startswith("# E1 final report")


def test_a_stored_eval_report_is_reused_only_from_the_same_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    cfg = "detector: yolov8n\ntracker: bytetrack\ninput_size: 640\nengine: ultralytics\n"
    (tmp_path / "configs" / "a.yaml").write_text(cfg + "stride: 1\nfoot_point: bottom\n")
    stored = {
        "source_sha256": cli.source_sha256(),
        "n_resamples": 1000,
        "configs": {"configs_a": vars(cli.load_config("configs/a.yaml"))},
    }
    path = tmp_path / "reports" / "eval-configs_a-tune-0.5m.json"
    cli.write_report(path, stored)
    computed = []
    monkeypatch.setattr(
        cli, "config_eval", lambda *a: computed.append(a) or ({"fresh": True}, None, ["configs_a"])
    )
    job = ("configs/a.yaml", None, "tune")

    reused = e1_final.config_evals([job], 0.5, 1000, print)[job]
    cli.write_report(path, stored | {"source_sha256": "other"})
    redone = e1_final.config_evals([job], 0.5, 1000, print)[job]

    assert reused == stored and redone == {"fresh": True} and len(computed) == 1
    assert json.loads(path.read_text()) == {"fresh": True}  # the new report is stored


def passing_inputs():
    """inputs() with every map target, the 20 % rule and the TVD confirmations met."""
    data = inputs()
    good = eval_report("sealed", 0.2, 0.05, 0.2, 1.0, 0.2, 0.2)
    data["evals"] = {"tune": good | {"split": "tune"}, "sealed": good}
    data["confirmations"] = data["confirmations"][:1]
    return data


def test_the_tracking_guard_is_checked_per_step_on_its_sealed_delta():
    s = e1_final.assemble(inputs())

    guards = [c["tracking_guard"] for c in s["confirmations"]]
    assert [g["holds"] for g in guards] == [True, False, True]
    assert round(guards[1]["hota_points"], 1) == 2.0 and round(guards[1]["idf1_points"], 1) == -1.2


def test_e1_is_accepted_only_when_every_step_keeps_the_tracking_guard():
    assert e1_final.assemble(passing_inputs())["e1_accepted"] is True

    broken = passing_inputs()
    broken["confirmations"][0]["sealed"] = eval_report(
        "sealed", 0, 0, 0, 0, 0, 0, (-0.191, -0.203, -0.18), (0.0, -0.011)
    )
    assert e1_final.assemble(broken)["e1_accepted"] is False


def test_hota_and_idf1_are_informational_against_tune_without_the_20_percent_verdict():
    s = e1_final.assemble(inputs())
    md = e1_final.render_markdown(s)

    assert all("within_20pct" not in x for x in s["tracking"].values())
    assert "| HOTA | 0.237 | 0.218 | +8 % | information only |" in md
    assert "Degradation (positive = worse)" in md
    assert "| ΔHOTA / ΔIDF1 sealed | Tracking guard |" in md
    assert "| +2.0 / -1.2 | broken |" in md
