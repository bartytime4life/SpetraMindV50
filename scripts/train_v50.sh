#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Training Wrapper
#
# Runs Hydra-safe training via the unified CLI with full logging, hashing, and safeguards.
#
# Usage:
#   bash scripts/train_v50.sh [--config path/to/config_v50.yaml] [--tag myrun] [--dry-run]
#   bash scripts/train_v50.sh --help
#
# ------------------------------------------------------------------------------------------
# Features:
# - Creates timestamped log dir under logs/
# - Passes Hydra config into CLI
# - Writes console logs and preserves exit code
# - Computes/records run hashes post-run
# - Optional --dry-run forwards to CLI dry run
# - Optional --tag attaches a human-friendly label to the output log dir
# - Auto-detects Python runner (uv/poetry/pip) but defaults to python
# ==========================================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_help() {
  cat <<'USAGE'
SpectraMind V50: train_v50.sh

USAGE:
  bash scripts/train_v50.sh [options]

OPTIONS:
  --config PATH     Hydra YAML path (default: configs/model/config_v50.yaml)
  --tag NAME        Tag for this run; appended to logs directory name
  --dry-run         Do not execute training, just validate configs and CLI routing
  --extra ARGS      Extra args to forward to the CLI (quote-encapsulate)
  --help            Show this help

EXAMPLES:
  bash scripts/train_v50.sh
  bash scripts/train_v50.sh --config configs/model/config_v50.yaml --tag v50_base
  bash scripts/train_v50.sh --dry-run --extra "+trainer.max_epochs=1"
USAGE
}

CONFIG="configs/model/config_v50.yaml"
TAG=""
DRY_RUN=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY_RUN="--dry-run"; shift 1 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

timestamp="$(date +"%Y%m%d_%H%M%S")"
base_log_dir="$REPO_ROOT/logs/train_${timestamp}"
[[ -n "$TAG" ]] && base_log_dir="${base_log_dir}_${TAG}"
mkdir -p "$base_log_dir"

# Python launcher detection
run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then
    poetry run python "$@"
  else
    python "$@"
  fi
}

# Sanity checks
if [[ ! -f "$REPO_ROOT/$CONFIG" ]]; then
  echo "[ERROR] Missing config file: $CONFIG"
  exit 3
fi

echo "[train_v50] repo: $REPO_ROOT"
echo "[train_v50] config: $CONFIG"
echo "[train_v50] logs: $base_log_dir"

set +e
run_py -m src.spectramind.spectramind train \
  +config="$CONFIG" \
  hydra.run.dir="$base_log_dir" \
  $DRY_RUN \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$base_log_dir/console.log"
code=$?
set -e

# Compute hash & write manifest
bash "$SCRIPT_DIR/hash_config.sh" "$base_log_dir" "$CONFIG" || true

if [[ $code -ne 0 ]]; then
  echo "[train_v50] Training failed with exit code $code"
  exit $code
fi
echo "[train_v50] Completed OK -> $base_log_dir"
