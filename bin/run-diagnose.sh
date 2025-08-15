#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate-env.sh"
echo "[DIAGNOSE] Generating diagnostics..."
python -m src.spectramind.cli.cli_diagnose dashboard "$@"
