#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate-env.sh"
echo "[TRAIN] Starting training..."
python -m src.spectramind.cli.cli_core_v50 train \
  --config-name=config_v50.yaml \
  "$@"
echo "[TRAIN] Completed at $(date)"
