"""The third-party contract of pyproject.toml is a frozen list: a package
installed later would be importable from domain unnoticed. This compares it
with the installed distributions, the way scripts/check-layers.sh lists them."""

import importlib.metadata as md
import keyword
import sys
import tomllib
from pathlib import Path

ALLOWED = {"numpy", "ot"}  # pure numerical libraries (docs/architecture.md)
CONTRACT = "domain and application import no third-party package but numpy and ot"


def test_every_installed_third_party_package_but_numpy_and_ot_is_forbidden():
    config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["importlinter"]
    (contract,) = [c for c in config["contracts"] if c["name"] == CONTRACT]
    own = set(config["root_packages"]) | set(sys.stdlib_module_names)
    installed = {
        name
        for name in md.packages_distributions()
        if name.isidentifier() and not name.startswith("_") and not keyword.iskeyword(name)
    }

    missing = installed - own - ALLOWED - set(contract["forbidden_modules"])

    assert not missing, f"add to the forbidden_modules of {CONTRACT!r}: {sorted(missing)}"
