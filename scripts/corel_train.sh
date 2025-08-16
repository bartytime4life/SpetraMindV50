#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - COREL Train Wrapper
#
# Trains COREL conformal/graph-based uncertainty calibrator with full logging.
#
# Usage:
#   bash scripts/corel_train.sh [--config configs/calibration/corel.yaml] [--extra "..."]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="configs/calibration/corel.yaml"
EXTRA_ARGS=()

print_help() {
  cat <<'USAGE'
SpectraMind V50: corel_train.sh

USAGE:
  bash scripts/corel_train.sh [options]

OPTIONS:
  --config PATH   Hydra config for COREL training
  --extra ARGS    Extra args (quoted)
  --help          Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

timestamp="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="$REPO_ROOT/logs/corel_${timestamp}"
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

set +e
run_py -m src.spectramind.spectramind corel-train \
  +config="$CONFIG" \
  outputs.dir="$OUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

echo "[corel_train] Done -> $OUT_DIR"
exit $code
