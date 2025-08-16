#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Push-Ready Check
#
# Ensures repo is clean, selftest passes (fast), and key configs exist prior to pushing.
#
# Usage:
#   bash scripts/push_ready_check.sh
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[push_ready_check] Checking git status…"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  echo "[push_ready_check] Working tree not clean. Commit or stash changes first."
  exit 3
fi

# Confirm critical configs exist
CRITICAL=(
  "configs/model/config_v50.yaml"
  "configs/diagnostics/dashboard.yaml"
  "configs/calibration/steps/pipeline.yaml"
)
for c in "${CRITICAL[@]}"; do
  if [[ ! -f "$c" ]]; then
    echo "[push_ready_check] Missing required config: $c"
    exit 4
  fi
done

# Run fast selftest
bash "$SCRIPT_DIR/ci_check.sh" --fast --no-lint || { echo "[push_ready_check] CI check failed"; exit 5; }

echo "[push_ready_check] Repository is push-ready."
