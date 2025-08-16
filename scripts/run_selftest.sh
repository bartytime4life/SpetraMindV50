#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Repository Self-Test Wrapper
#
# Verifies CLI registration, config presence, symbolic routing, shapes, and quick diagnostics.
#
# Usage:
#   bash scripts/run_selftest.sh [--deep] [--open-html] [--clean]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEEP=0
OPEN_HTML=0
CLEAN=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: run_selftest.sh

USAGE:
  bash scripts/run_selftest.sh [options]

OPTIONS:
  --deep        Run deep mode (more thorough checks; slower)
  --open-html   Open generated HTML diagnostics (if present)
  --clean       Run log dedupe/cleanup post-test
  --help        Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deep) DEEP=1; shift ;;
    --open-html) OPEN_HTML=1; shift ;;
    --clean) CLEAN=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

timestamp="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="$REPO_ROOT/logs/selftest_${timestamp}"
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

EXTRA=()
[[ $DEEP -eq 1 ]] && EXTRA+=("--deep")
[[ $OPEN_HTML -eq 1 ]] && EXTRA+=("--open-html")

set +e
run_py "$REPO_ROOT/selftest.py" --out "$OUT_DIR" "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

if [[ $CLEAN -eq 1 ]]; then
  bash "$SCRIPT_DIR/clean_logs.sh" --dedupe --keep 30 || true
fi

exit $code
