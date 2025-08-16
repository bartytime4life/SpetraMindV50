#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Full Calibration Pipeline Wrapper
#
# Runs pre-processing calibration and uncertainty calibration (temperature scaling + COREL).
#
# Usage:
#   bash scripts/calibrate_v50.sh [--config configs/calibration/steps/pipeline.yaml] [--extra "..."]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_help() {
  cat <<'USAGE'
SpectraMind V50: calibrate_v50.sh

USAGE:
  bash scripts/calibrate_v50.sh [options]

OPTIONS:
  --config PATH   Hydra config for calibration pipeline (default: configs/calibration/steps/pipeline.yaml)
  --extra ARGS    Extra args forwarded to CLI (quote them)
  --help          Show help
USAGE
}

CONFIG="configs/calibration/steps/pipeline.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

timestamp="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="$REPO_ROOT/logs/calibration_${timestamp}"
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

set +e
run_py -m src.spectramind.spectramind calibrate \
  +config="$CONFIG" \
  outputs.dir="$OUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

bash "$SCRIPT_DIR/hash_config.sh" "$OUT_DIR" "$CONFIG" || true
exit $code
