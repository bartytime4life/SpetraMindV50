#!/usr/bin/env bash

# Sync conf/hydra to mirror configs/hydra using the canonical checker.

# Optional flags are passed through (e.g., --prune, --normalize).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
if [ ! -f tools/check_hydra_sync.py ]; then
  echo "[ERROR] tools/check_hydra_sync.py not found. Run the bootstrap installer first." >&2
  exit 1
fi
python tools/check_hydra_sync.py --fix "$@"
echo "Hydra conf tree synced to configs ✅"
