#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Local CI Check
#
# Runs unit tests, selftest (fast mode), Hydra config validation, and lint (if available).
#
# Usage:
#   bash scripts/ci_check.sh [--fast] [--open-html] [--no-lint]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FAST=0
OPEN_HTML=0
NO_LINT=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: ci_check.sh

USAGE:
  bash scripts/ci_check.sh [options]

OPTIONS:
  --fast        Faster checks (skips heavy tests)
  --open-html   Open HTML diagnostics after selftest if present
  --no-lint     Skip lint/format checks
  --help        Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1; shift ;;
    --open-html) OPEN_HTML=1; shift ;;
    --no-lint) NO_LINT=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

echo "[ci_check] Running unit tests…"
if command -v pytest >/dev/null 2>&1; then
  if [[ $FAST -eq 1 ]]; then
    pytest -q || { echo "[ci_check] pytest failed"; exit 10; }
  else
    pytest -q --maxfail=1 || { echo "[ci_check] pytest failed"; exit 10; }
  fi
else
  echo "[ci_check] pytest not found; skipping tests."
fi

echo "[ci_check] Running selftest…"
EXTRA=()
[[ $FAST -eq 1 ]] && EXTRA+=("--fast")
OUT_DIR="$REPO_ROOT/logs/ci_selftest_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$OUT_DIR"
set +e
run_py "$REPO_ROOT/selftest.py" --out "$OUT_DIR" "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e
if [[ $code -ne 0 ]]; then
  echo "[ci_check] selftest failed with code $code"
  exit $code
fi

if [[ $NO_LINT -eq 0 ]]; then
  echo "[ci_check] Running lint/format (best-effort)…"
  if command -v ruff >/dev/null 2>&1; then ruff check src || true; fi
  if command -v black >/dev/null 2>&1; then black --check src || true; fi
fi

if [[ $OPEN_HTML -eq 1 ]]; then
  bash "$SCRIPT_DIR/launch_dashboard.sh" --latest || true
fi

echo "[ci_check] OK"
