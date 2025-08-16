#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Environment Snapshot Export
#
# Exports environment specs for reproducibility (pip freeze, Poetry export, Conda env, etc.).
#
# Usage:
#   bash scripts/export_env.sh [--out .envsnap] [--poetry] [--conda] [--pip]
# ==========================================================================================

set -Eeuo pipefail
OUT_DIR=".envsnap"
USE_POETRY=0
USE_CONDA=0
USE_PIP=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: export_env.sh

USAGE:
  bash scripts/export_env.sh [options]

OPTIONS:
  --out DIR     Output directory (default: .envsnap)
  --poetry      Export Poetry lock as requirements.txt (poetry export)
  --conda       Export Conda environment (conda env export)
  --pip         Export pip freeze
  --help        Show help

NOTE: You can specify multiple exporters (e.g., --poetry --pip).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --poetry) USE_POETRY=1; shift ;;
    --conda) USE_CONDA=1; shift ;;
    --pip) USE_PIP=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ $USE_POETRY -eq 1 ]]; then
  if command -v poetry >/dev/null 2>&1; then
    poetry export -f requirements.txt --output "$OUT_DIR/requirements_from_poetry.txt" --without-hashes || true
  else
    echo "[export_env] poetry not available; skipping poetry export."
  fi
fi

if [[ $USE_CONDA -eq 1 ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda env export > "$OUT_DIR/conda_environment.yaml" || true
  else
    echo "[export_env] conda not available; skipping conda export."
  fi
fi

if [[ $USE_PIP -eq 1 ]]; then
  python -m pip freeze > "$OUT_DIR/pip_freeze.txt" || true
fi

# Always capture Python & CUDA info
{
  echo "===== PYTHON VERSION ====="
  python -c 'import sys,platform;print(sys.version);print(platform.platform())' || true
  echo "===== CUDA (nvidia-smi) ====="
  nvidia-smi || true
} > "$OUT_DIR/system_info.txt"

echo "[export_env] Snapshot written to $OUT_DIR"
