"""ultralytics is AGPL-3.0: only evaluation/baseline imports it, and only the
CLI imports evaluation/baseline, inside the commands that need it."""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path("src/footfall_heatmaps")
BASELINE = PACKAGE / "evaluation" / "baseline"
AGPL = ("ultralytics",)
BASELINE_MODULE = "footfall_heatmaps.evaluation.baseline"


def imports(path):
    """(module, at top level) for every import statement in a file."""
    tree = ast.parse(path.read_text())
    top = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((a.name, id(node) in top) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, id(node) in top


def under(module, root):
    return module == root or module.startswith(root + ".")


def test_only_the_baseline_package_imports_ultralytics():
    offenders = [
        (str(f), m)
        for f in PACKAGE.rglob("*.py")
        if BASELINE not in f.parents
        for m, _ in imports(f)
        if any(under(m, a) for a in AGPL)
    ]
    assert not offenders


def test_only_the_cli_imports_the_baseline_and_never_at_import_time():
    offenders = [
        (str(f), m, top)
        for f in PACKAGE.rglob("*.py")
        if BASELINE not in f.parents
        for m, top in imports(f)
        if under(m, BASELINE_MODULE) and (f != PACKAGE / "cli.py" or top)
    ]
    assert not offenders


def test_importing_the_cli_loads_no_agpl_module():
    code = (
        "import sys, footfall_heatmaps.cli; "
        f"print([m for m in sys.modules if m.split('.')[0] in {AGPL!r}])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
