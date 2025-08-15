#!/usr/bin/env bash
set -euo pipefail
echo "[DEPS] Updating dependencies..."
poetry install
poetry update
echo "[DEPS] Completed at $(date)"
