#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate-env.sh"
echo "[SELFTEST] Running pipeline selftest..."
python -m src.spectramind.selftest --mode fast
