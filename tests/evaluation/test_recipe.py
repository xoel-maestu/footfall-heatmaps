import re
import subprocess
import sys

from footfall_heatmaps.evaluation.baseline.recipe import recipe_metadata
from footfall_heatmaps.evaluation.domain.run_config import RunConfig

CONFIG = RunConfig("yolov8n", "bytetrack", 640, "centroid", "ultralytics", 1)


def test_the_metadata_names_everything_the_tracks_depend_on():
    m = recipe_metadata(CONFIG)

    assert set(m) == {
        "ultralytics", "tracker_yaml_sha256", "classes", "conf", "imgsz", "weights", "stride",
    }  # fmt: skip
    assert (m["classes"], m["conf"], m["imgsz"], m["weights"]) == ([0], 0.1, 640, "yolov8n.pt")
    # Installed (uv sync --group baseline): the version and the tracker yaml hash;
    # not installed (CI): unknown, None.
    assert (m["ultralytics"] is None) == (m["tracker_yaml_sha256"] is None)
    if m["tracker_yaml_sha256"] is not None:
        assert re.fullmatch(r"[0-9a-f]{64}", m["tracker_yaml_sha256"])


def test_reading_the_metadata_does_not_import_ultralytics():
    code = (
        "import sys\n"
        "from footfall_heatmaps.evaluation.baseline.recipe import recipe_metadata\n"
        "from footfall_heatmaps.evaluation.domain.run_config import RunConfig\n"
        "recipe_metadata(RunConfig('yolov8n', 'bytetrack', 640, 'centroid', 'ultralytics', 1))\n"
        "print([m for m in sys.modules if m.split('.')[0] == 'ultralytics'])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
