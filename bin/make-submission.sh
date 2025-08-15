#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate-env.sh"
echo "[SUBMISSION] Running selftest..."
bin/selftest.sh
echo "[SUBMISSION] Training..."
bin/run-train.sh "$@"
echo "[SUBMISSION] Predicting..."
bin/run-predict.sh "$@"
echo "[SUBMISSION] Validating + Bundling..."
python -m src.spectramind.cli.cli_submit make-submission "$@"
