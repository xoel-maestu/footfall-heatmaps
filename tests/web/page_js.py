"""Run a marked block of the viewer page (web/static/index.html) in Node.

Without node the tests are skipped locally; in CI (the CI variable set) node is
required, so a missing node fails the run instead of skipping it silently."""

import json
import os
import re
import shutil
import subprocess

import pytest

from footfall_heatmaps.web.server import PAGE

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(
    NODE is None and not os.environ.get("CI"), reason="node is not installed"
)


def page_block(name):
    block = re.search(rf"// {name}:begin\n(.*?)// {name}:end", PAGE.read_text(), re.S)
    assert block, f"index.html has no block between the {name} markers"
    return block.group(1)


def run_js(blocks, expression):
    """The JSON value of `expression` after the page's named blocks."""
    assert NODE, "node is required when CI is set"
    script = "\n".join(page_block(b) for b in blocks)
    script += f"\nconsole.log(JSON.stringify({expression}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def js(value):
    return json.dumps(value)
