#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Generate Dummy Test Data
#
# Produces synthetic FGS1/AIRS cubes and minimal metadata for end-to-end pipeline checks.
#
# Usage:
#   bash scripts/generate_dummy_data.sh [--out data/dummy] [--n 10] [--seed 42] [--extra "..."]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="data/dummy"
N=10
SEED=42
EXTRA_ARGS=()

print_help() {
  cat <<'USAGE'
SpectraMind V50: generate_dummy_data.sh

USAGE:
  bash scripts/generate_dummy_data.sh [options]

OPTIONS:
  --out DIR      Output directory for generated data (default: data/dummy)
  --n N          Number of planets/samples (default: 10)
  --seed S       Random seed (default: 42)
  --extra ARGS   Extra args forwarded to CLI (quote them)
  --help         Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

set +e
run_py -m src.spectramind.spectramind generate-dummy-data \
  outputs.dir="$OUT_DIR" \
  generator.n="$N" generator.seed="$SEED" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

echo "[generate_dummy_data] Done -> $OUT_DIR"
exit $code
