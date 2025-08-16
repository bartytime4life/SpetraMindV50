#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - UMAP Latent Visualization
#
# Wrapper for UMAP plotting with symbolic overlays and HTML dashboard embedding.
#
# Usage:
#   bash scripts/umap_latents.sh [--config configs/diagnostics/umap.yaml] [--open]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="configs/diagnostics/umap.yaml"
OPEN=0
EXTRA_ARGS=()

print_help() {
  cat <<'USAGE'
SpectraMind V50: umap_latents.sh

USAGE:
  bash scripts/umap_latents.sh [options]

OPTIONS:
  --config PATH   Hydra config for UMAP latent visualization
  --open          Open resulting HTML
  --extra ARGS    Extra args (quoted)
  --help          Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --open) OPEN=1; shift ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

timestamp="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="$REPO_ROOT/logs/umap_${timestamp}"
mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then uv run python "$@"
  elif command -v poetry >/dev/null 2>&1 && poetry env info >/dev/null 2>&1; then poetry run python "$@"
  else python "$@"; fi
}

set +e
run_py -m src.spectramind.spectramind diagnose umap-latents \
  +config="$CONFIG" \
  outputs.dir="$OUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/console.log"
code=$?
set -e

if [[ $OPEN -eq 1 ]]; then
  bash "$SCRIPT_DIR/launch_dashboard.sh" --latest || true
fi
exit $code
