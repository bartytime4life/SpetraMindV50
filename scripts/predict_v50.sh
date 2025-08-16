#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Prediction & Packaging Wrapper
#
# Produces μ/σ predictions and optional submission bundle; integrates calibration if desired.
#
# Usage:
#   bash scripts/predict_v50.sh [--config configs/model/config_v50.yaml] [--ckpt path]
#                                [--out outdir] [--bundle] [--open-html]
#                                [--extra "..."]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_help() {
  cat <<'USAGE'
SpectraMind V50: predict_v50.sh

USAGE:
  bash scripts/predict_v50.sh [options]

OPTIONS:
  --config PATH      Hydra config (default: configs/model/config_v50.yaml)
  --ckpt PATH        Model checkpoint to load (required for inference)
  --out DIR          Output directory (default: logs/predict_TIMESTAMP[_TAG])
  --bundle           Create Kaggle submission zip after prediction
  --open-html        Open diagnostics HTML if generated
  --extra ARGS       Extra args forwarded to CLI (quote them)
  --help             Show help

EXAMPLES:
  bash scripts/predict_v50.sh --ckpt artifacts/v50.ckpt --bundle
  bash scripts/predict_v50.sh --config configs/model/config_v50.yaml --ckpt ckpts/best.pt --extra "+inference.batch_size=32"
USAGE
}

CONFIG="configs/model/config_v50.yaml"
CKPT=""
OUT_DIR=""
BUNDLE=0
OPEN_HTML=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --bundle) BUNDLE=1; shift 1 ;;
    --open-html) OPEN_HTML=1; shift 1 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "[ERROR] --ckpt is required."
  exit 3
fi

timestamp="$(date +"%Y%m%d_%H%M%S")"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/logs/predict_${timestamp}"
fi
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then
    poetry run python "$@"
  else
    python "$@"
  fi
}

# Prediction
set +e
run_py -m src.spectramind.spectramind predict \
  +config="$CONFIG" \
  inference.ckpt="$CKPT" \
  outputs.dir="$OUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

bash "$SCRIPT_DIR/hash_config.sh" "$OUT_DIR" "$CONFIG" || true
if [[ $code -ne 0 ]]; then
  echo "[predict_v50] Prediction failed with exit code $code"
  exit $code
fi

# Optional bundling
if [[ $BUNDLE -eq 1 ]]; then
  bash "$SCRIPT_DIR/bundle_submission.sh" --in "$OUT_DIR" --open-html=$OPEN_HTML
fi
echo "[predict_v50] Done -> $OUT_DIR"
