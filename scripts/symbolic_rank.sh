#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Symbolic Rule Ranking
#
# Wrapper for spectramind diagnose symbolic-rank with exports.
#
# Usage:
#   bash scripts/symbolic_rank.sh [--config configs/diagnostics/symbolic_rank.yaml] [--extra "..."]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="configs/diagnostics/symbolic_rank.yaml"
EXTRA_ARGS=()

print_help() {
  cat <<'USAGE'
SpectraMind V50: symbolic_rank.sh

USAGE:
  bash scripts/symbolic_rank.sh [options]

OPTIONS:
  --config PATH   Hydra config for symbolic rank
  --extra ARGS    Extra CLI args (quoted)
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
OUT_DIR="$REPO_ROOT/logs/symbolic_rank_${timestamp}"
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

set +e
run_py -m src.spectramind.spectramind diagnose symbolic-rank \
  +config="$CONFIG" \
  outputs.dir="$OUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

echo "[symbolic_rank] Finished -> $OUT_DIR"
exit $code
