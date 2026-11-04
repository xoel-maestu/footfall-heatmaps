#!/usr/bin/env bash
# The only gate of this repository. CI runs it, and so do you.
#
# Python with uv: once there is a pyproject.toml it syncs the environment and
# runs ruff and pytest, then the hexagonal layer check (scripts/check-layers.sh).
# Passing this is a prerequisite, not a success criterion.
set -euo pipefail
cd "$(dirname "$0")/.."

# Local tooling that is not part of the published repo. Absent in CI and in a
# fresh clone: skipped, never a failure.
if [ -x scripts/harness-check.sh ]; then
  ./scripts/harness-check.sh
else
  echo "== harness =="
  echo "(scripts/harness-check.sh not present in this checkout: skipped)"
fi

if [ -f pyproject.toml ]; then
  echo "== python (uv) =="
  if ! command -v uv >/dev/null; then
    echo "pyproject.toml exists but uv is not installed: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  if [ -f uv.lock ]; then
    uv sync --locked
  else
    uv sync
  fi
  uv run --no-sync ruff check .
  uv run --no-sync ruff format --check .
  # Exit code 5 is "no tests collected": allowed until the first test exists.
  rc=0
  uv run --no-sync pytest -q || rc=$?
  if [ "$rc" = 5 ]; then
    echo "WARNING: pytest collected no tests."
  elif [ "$rc" != 0 ]; then
    exit "$rc"
  fi
else
  echo "== python =="
  echo "(no pyproject.toml yet: uv, ruff and pytest skipped)"
fi

# domain <- application <- infrastructure (import-linter from .venv).
./scripts/check-layers.sh
