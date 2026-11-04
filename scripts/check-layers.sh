#!/usr/bin/env bash
# managed-by: plantilla. Edit it there: scripts/harness-sync.sh overwrites this copy.
# Layer check of the hexagonal layout (docs/architecture/hexagonal.md):
# infrastructure -> application -> domain, never the other way round.
# verify.sh calls it.
#
# It finds each stack by its layout and runs the tool that fits:
#   TypeScript  package.json + src/contexts/                  dependency-cruiser
#   Python      pyproject.toml + a package with domain/       import-linter
#   Rust        a crate with [package.metadata.hexagonal]     cargo metadata
#               role = "core" (Cargo.toml or src-tauri/Cargo.toml)
# A stack without that layout yet is skipped, and the script says so. A tool
# that is needed but missing is a failure with the install command, never a
# silent pass.
#
# The rules are generated here, so every repo starts with the same ones and
# this file can be synced from the template. A repo that needs other rules
# writes its own config (.dependency-cruiser.{cjs,mjs,js,json,json5},
# .importlinter or [tool.importlinter]), and then that one is used instead.
# To start one from the generated rules:
#   scripts/check-layers.sh --print-config ts|py
set -euo pipefail
cd "$(dirname "$0")/.."

print=""
if [ "${1:-}" = --print-config ]; then
  case "${2:-}" in
    ts|py) print="$2" ;;
    *) echo "usage: $0 [--print-config ts|py]" >&2; exit 2 ;;
  esac
fi

[ -n "$print" ] || echo "== layers =="
status=0
checked=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

violation() {
  echo "Layer violation ($1). The dependency rule: docs/architecture/hexagonal.md." >&2
  status=1
}

# --- TypeScript: dependency-cruiser ---------------------------------------
# Domain imports only domain (any context's) and application only domain and
# application: no infrastructure, no src/app, no src/lib or other code outside
# the contexts, which may do I/O. Neither imports an npm package or a Node core
# module. Type-only imports count (tsPreCompilationDeps): a port is often
# imported only as a type. An import it cannot resolve (a path alias it does
# not know, a package not installed) would go unchecked, so it is an error.
ts_config() {
  local tsconfig="" f
  # tsconfig.app.json first: with project references (the Vite layout) the
  # root tsconfig.json has no paths and @/ aliases would not resolve.
  for f in tsconfig.app.json tsconfig.json; do
    if [ -f "$f" ]; then tsconfig=", \"tsConfig\": {\"fileName\": \"$f\"}"; break; fi
  done
  cat <<JSON
{
  "forbidden": [
    {"name": "domain-imports-only-domain", "severity": "error",
     "from": {"path": "^src/contexts/[^/]+/domain/"},
     "to": {"path": "^src/", "pathNot": "^src/contexts/[^/]+/domain/"}},
    {"name": "application-imports-only-domain-and-application", "severity": "error",
     "from": {"path": "^src/contexts/[^/]+/application/"},
     "to": {"path": "^src/", "pathNot": "^src/contexts/[^/]+/(domain|application)/"}},
    {"name": "domain-and-application-do-no-io", "severity": "error",
     "from": {"path": "^src/contexts/[^/]+/(domain|application)/"},
     "to": {"dependencyTypes": ["core", "npm", "npm-dev", "npm-optional",
                                "npm-peer", "npm-bundled", "npm-no-pkg", "npm-unknown"]}},
    {"name": "contexts-do-not-import-app", "severity": "error",
     "from": {"path": "^src/contexts/"},
     "to": {"path": "^src/app/"}},
    {"name": "contexts-imports-resolve", "severity": "error",
     "comment": "Unresolved, so unchecked: an alias needs the tsconfig that defines it, a package npm install.",
     "from": {"path": "^src/contexts/"},
     "to": {"couldNotResolve": true}}
  ],
  "options": {"doNotFollow": {"path": "node_modules"}, "tsPreCompilationDeps": true$tsconfig}
}
JSON
}
if [ "$print" = ts ]; then ts_config; exit 0; fi

if [ -f package.json ]; then
  checked=1
  if [ ! -d src/contexts ]; then
    echo "(no src/contexts yet: TypeScript layer check skipped)"
  elif [ ! -x node_modules/.bin/depcruise ]; then
    echo "src/contexts exists but dependency-cruiser is not installed:" >&2
    echo "  npm install --save-dev dependency-cruiser" >&2
    status=1
  else
    config=""
    for f in .dependency-cruiser.{cjs,mjs,js,json,json5}; do
      if [ -f "$f" ]; then config="$f"; break; fi
    done
    if [ -z "$config" ]; then
      config="$tmp/dependency-cruiser.json"
      ts_config >"$config"
    fi
    echo "dependency-cruiser, rules from ${config#"$tmp/"}"
    if ! node_modules/.bin/depcruise src --config "$config" --output-type err | tee "$tmp/depcruise.out"; then
      violation "dependency-cruiser"
    elif grep -qE '(^|[^0-9])0 modules' "$tmp/depcruise.out"; then
      # Without a TypeScript compiler it supports, it reads nothing and passes.
      {
        echo "dependency-cruiser read no module under src/: that is not a pass."
        echo "  Usual cause: a TypeScript it does not support. Releases without TypeScript 7"
        echo "  support read 0 modules with it ('npx depcruise --info' lists what it supports"
        echo "  and what is installed). Fix: npm install --save-dev typescript@6, until your"
        echo "  dependency-cruiser declares TypeScript 7 in its peerDependencies"
        echo "  (docs/architecture/hexagonal.md)."
      } >&2
      status=1
    fi
  fi
fi

# --- Python: import-linter ------------------------------------------------
# A context is any package with a domain/ subpackage, inside a top-level
# package (<pkg>/ or src/<pkg>/, not tests/): <pkg>/<context>/domain/ or
# deeper, like <pkg>/contexts/<context>/domain/. Every level has __init__.py.
# Domain imports neither application nor infrastructure (of any context);
# application does not import infrastructure; neither imports a module of the
# package that is outside the contexts (<pkg>/db.py, <pkg>/web/); and neither
# imports a third-party package. Third-party means every top-level module of
# an installed distribution, read with importlib.metadata from the Python that
# runs lint-imports (its shebang, else the python beside it), since that is the
# environment import-linter resolves imports in: that maps distribution names
# to import names (PyYAML -> yaml), which pyproject.toml does not.
dotted() { local m="${1#src/}"; m="${m%.py}"; printf '%s' "${m//\//.}"; }
if [ -f pyproject.toml ] || [ -f requirements.txt ] || [ "$print" = py ]; then
  checked=1
  contexts=()
  for init in */__init__.py src/*/__init__.py; do
    root="${init%/__init__.py}"
    case "$root" in tests | test | src/tests | src/test) continue ;; esac
    [ -f "$init" ] || continue
    while IFS= read -r f; do contexts+=("${f%/domain/__init__.py}"); done \
      < <(find "$root" -path '*/domain/__init__.py' | sort)
  done
  own=0
  if [ -f .importlinter ] || grep -qs '^\[tool\.importlinter\]' pyproject.toml \
     || grep -qs '^\[importlinter\]' setup.cfg; then
    own=1
  fi
  lint="lint-imports"
  [ -x .venv/bin/lint-imports ] && lint=.venv/bin/lint-imports

  # Modules per layer, one per line, as dotted names:
  # src/shop/orders/domain -> shop.orders.domain. "outside" is every module
  # beside a context or one of its parent packages that holds no context.
  roots="" domain="" application="" infrastructure="" outside=""
  for c in "${contexts[@]}"; do
    m="$(dotted "$c")"
    roots+="    ${m%%.*}"$'\n'
    for l in domain application infrastructure; do
      if [ -e "$c/$l" ] || [ -e "$c/$l.py" ]; then
        printf -v "$l" '%s    %s.%s\n' "${!l}" "$m" "$l"
      fi
    done
    a="$c"
    while [ "$(dotted "$a")" != "${m%%.*}" ]; do
      a="${a%/*}"
      for e in "$a"/*; do
        case "$e" in
          */__init__.py) continue ;;
          *.py) ;;
          *) [ -f "$e/__init__.py" ] || continue ;;
        esac
        for x in "${contexts[@]}"; do [[ "$x/" == "${e%.py}/"* ]] && continue 2; done
        outside+="    $(dotted "$e")"$'\n'
      done
    done
  done
  outside="$(printf '%s' "$outside" | sort -u)"
  [ -z "$outside" ] || outside+=$'\n'
  # The Python lint-imports runs with: a venv, a uv tool, pipx or poetry each
  # have their own, and python3 on PATH may have none of the project's packages.
  lint_python() {
    local p line
    p="$(command -v "$lint")" || return 1
    IFS= read -r line <"$p" || true
    line="${line#\#!}"
    # shellcheck disable=SC2086  # the shebang's words
    set -- $line
    [ "${1##*/}" = env ] && shift
    case "${1:-}" in */python* | python*) command -v "$1" && return 0 ;; esac
    [ -x "${p%/*}/python" ] && printf '%s' "${p%/*}/python"
  }
  external="" external_from=""
  python="$(lint_python || true)"
  if [ "${#contexts[@]}" -gt 0 ] && [ -n "$python" ]; then
    # shellcheck disable=SC2086  # $roots: one package name per word
    if external="$("$python" - $roots <<'PY'
import importlib.metadata as md, keyword, sys
stdlib = getattr(sys, "stdlib_module_names", None)
if stdlib is None:
    sys.exit(f"Python {sys.version.split()[0]} has no sys.stdlib_module_names (3.10 or later needed): run import-linter from a newer Python.")
own = set(sys.argv[1:]) | set(stdlib)
for name in sorted(md.packages_distributions()):
    if name.isidentifier() and not name.startswith("_") and not keyword.iskeyword(name) and name not in own:
        print("    " + name)
PY
)"; then
      external_from="$(grep -c . <<<"$external" || true) third-party modules forbidden in domain and application, from the distributions installed in $python"
    else
      echo "Could not list the distributions installed in $python (above): third-party imports not checked." >&2
      external="" status=1
    fi
    [ -z "$external" ] || external+=$'\n'
  fi
  ini="$tmp/importlinter.ini"
  if [ -n "$application$infrastructure$outside$external" ]; then
    {
      printf '[importlinter]\nroot_packages =\n%s\n' "$(printf '%s' "$roots" | sort -u)"
      [ -z "$external" ] || printf 'include_external_packages = True\n'
      if [ -n "$application$infrastructure$outside" ]; then
        printf '\n[importlinter:contract:domain]\nname = domain imports only domain\n'
        printf 'type = forbidden\nsource_modules =\n%sforbidden_modules =\n%s%s%s' \
          "$domain" "$application" "$infrastructure" "$outside"
      fi
      if [ -n "$application" ] && [ -n "$infrastructure$outside" ]; then
        printf '\n[importlinter:contract:application]\nname = application imports only domain and application\n'
        printf 'type = forbidden\nsource_modules =\n%sforbidden_modules =\n%s%s' \
          "$application" "$infrastructure" "$outside"
      fi
      if [ -n "$external" ]; then
        printf '\n[importlinter:contract:external]\nname = domain and application import no third-party package\n'
        printf 'type = forbidden\nsource_modules =\n%s%sforbidden_modules =\n%s' \
          "$domain" "$application" "$external"
      fi
    } >"$ini"
  fi
  if [ "$print" = py ]; then
    [ -n "$python" ] || echo "(no lint-imports found, so no Python to list third-party packages from: that contract is left out)" >&2
    if [ -f "$ini" ]; then cat "$ini"; else echo "(no Python context with something to check: nothing to print)" >&2; fi
    exit 0
  fi

  if [ "${#contexts[@]}" = 0 ] && [ "$own" = 0 ]; then
    echo "(no package with a domain/ subpackage yet: Python layer check skipped)"
  elif ! command -v "$lint" >/dev/null; then
    echo "There are Python contexts but import-linter is not installed:" >&2
    echo "  pip install import-linter   (or: uv add --dev import-linter)" >&2
    status=1
  elif [ "$own" = 1 ]; then
    echo "import-linter, this repo's contracts"
    PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}" "$lint" || violation "import-linter"
  elif [ ! -f "$ini" ]; then
    echo "(Python contexts have only a domain layer: nothing to check yet)"
  else
    echo "import-linter, generated contracts for: ${contexts[*]}"
    if [ -n "$external_from" ]; then
      echo "  $external_from"
    elif [ -z "$python" ]; then
      echo "Could not tell which Python runs $(command -v "$lint") (not a python shebang, no python beside it): third-party imports not checked." >&2
      status=1
    fi
    # --no-cache: its cache does not tell graphs with and without external
    # packages apart, so a module cached by a run without them hides its
    # third-party imports.
    PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}" "$lint" --no-cache --config "$ini" \
      || violation "import-linter"
  fi
fi

# --- Rust: cargo metadata -------------------------------------------------
# The core crate (domain + application) is the one marked in its Cargo.toml:
#   [package.metadata.hexagonal]
#   role = "core"
#   io_crates = ["some-db-driver"]   # optional, added to the list below
# It must not depend on a workspace member that is not core (an adapter, or a
# pure crate not marked core) nor on an I/O crate. Direct normal and build
# dependencies; dev ones are free. Tauri keeps its workspace in src-tauri/.
cargo_toml=""
for f in Cargo.toml src-tauri/Cargo.toml; do
  if [ -f "$f" ]; then cargo_toml="$f"; break; fi
done
if [ -n "$cargo_toml" ]; then
  checked=1
  # rustup installs into ~/.cargo/bin, which a non-interactive shell (ssh, CI,
  # an agent's tool call) often does not have on its PATH.
  if ! command -v cargo >/dev/null && [ -x "${CARGO_HOME:-$HOME/.cargo}/bin/cargo" ]; then
    PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  fi
  if ! command -v cargo >/dev/null; then
    echo "There is a $cargo_toml but cargo is not installed (not on PATH nor in ${CARGO_HOME:-$HOME/.cargo}/bin): https://rustup.rs" >&2
    status=1
  elif ! command -v python3 >/dev/null; then
    echo "There is a $cargo_toml: the Rust layer check reads cargo metadata with python3, which is not installed." >&2
    status=1
  else
    cargo metadata --format-version 1 --no-deps --manifest-path "$cargo_toml" >"$tmp/metadata.json"
    python3 - "$tmp/metadata.json" <<'PY' || violation "cargo metadata"
import json, sys

IO_CRATES = {
    "tokio", "async-std", "smol", "mio", "reqwest", "ureq", "hyper", "axum",
    "actix-web", "rocket", "warp", "tonic", "sqlx", "diesel", "rusqlite",
    "postgres", "tokio-postgres", "redis", "mongodb", "tauri",
}

meta = json.load(open(sys.argv[1]))
members = {p["name"]: p for p in meta["packages"] if p["id"] in meta["workspace_members"]}
hexagonal = {n: (p.get("metadata") or {}).get("hexagonal") or {} for n, p in members.items()}
cores = sorted(n for n, h in hexagonal.items() if h.get("role") == "core")
if not cores:
    print('(no crate with [package.metadata.hexagonal] role = "core": Rust layer check skipped)')
    sys.exit(0)

errors = []
for core in cores:
    forbidden = IO_CRATES | set(hexagonal[core].get("io_crates", []))
    for dep in members[core]["dependencies"]:
        name = dep["name"]
        if dep.get("kind") == "dev":
            continue
        if name in members and name not in cores:
            errors.append(f"{core} depends on {name}, a workspace crate not marked core: adapters depend on the core, never the reverse (a pure shared crate is marked role = \"core\" too).")
        elif name in forbidden:
            errors.append(f"{core} depends on {name}, an I/O crate: I/O belongs in an adapter crate.")
print("cargo metadata, core crates: " + ", ".join(cores), flush=True)
for e in errors:
    print("  " + e, file=sys.stderr)
sys.exit(1 if errors else 0)
PY
  fi
fi

[ "$checked" = 1 ] || echo "(no package.json, pyproject.toml, Cargo.toml or src-tauri/Cargo.toml: layer check skipped)"
exit "$status"
