#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Check CLI Command-to-File Map
#
# Generates a live page or JSON mapping of CLI commands to implementing files and exports.
#
# Usage:
#   bash scripts/check_cli_map.sh [--out logs/cli_map_TIMESTAMP] [--open]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR=""
OPEN=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: check_cli_map.sh

USAGE:
  bash scripts/check_cli_map.sh [options]

OPTIONS:
  --out DIR    Output directory (default: logs/cli_map_TIMESTAMP)
  --open       Open resulting HTML
  --help       Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --open) OPEN=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

timestamp="$(date +"%Y%m%d_%H%M%S")"
[[ -z "$OUT_DIR" ]] && OUT_DIR="$REPO_ROOT/logs/cli_map_${timestamp}"
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

set +e
run_py -m src.spectramind.spectramind analyze-log check-cli-map \
  outputs.dir="$OUT_DIR" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

if [[ $OPEN -eq 1 ]]; then
  bash "$SCRIPT_DIR/launch_dashboard.sh" --latest || true
fi
exit $code
