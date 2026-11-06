from space_usage_heatmaps import cli
from space_usage_heatmaps.evaluation.domain.splits import SPLITS


def test_sealed_split_is_scenes_076_to_080_one_camera_each():
    # Each camera: the scene's calibrated camera with the most ground_truth.txt rows,
    # the rule that picked the tune cameras.
    assert SPLITS["sealed"] == (
        ("076", "723"),
        ("077", "734"),
        ("078", "753"),
        ("079", "777"),
        ("080", "784"),
    )


def test_sealed_and_tune_share_no_scene():
    tune = {scene for scene, _ in SPLITS["tune"]}
    sealed = {scene for scene, _ in SPLITS["sealed"]}
    assert tune.isdisjoint(sealed)


def test_eval_accepts_split_sealed(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["split"] = args.split
        return 0

    monkeypatch.setattr(cli, "run_eval_config", fake_run)
    assert cli.main(["eval", "--config", "c.yaml", "--split", "sealed"]) == 0
    assert seen["split"] == "sealed"
