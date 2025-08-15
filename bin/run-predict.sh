#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate-env.sh"
echo "[PREDICT] Running prediction..."
python -m src.spectramind.cli.cli_core_v50 predict \
  --config-name=config_v50.yaml \
  "$@"
echo "[PREDICT] Finished at $(date)"
