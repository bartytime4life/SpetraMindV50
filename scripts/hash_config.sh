#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Config & Run Hashing
#
# Computes a stable hash for the run config and environment; writes run_hash_summary_v50.json
#
# Usage:
#   bash scripts/hash_config.sh OUT_DIR CONFIG_PATH
# ==========================================================================================

set -Eeuo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/hash_config.sh OUT_DIR CONFIG_PATH"
  exit 2
fi

OUT_DIR="$1"
CONFIG_PATH="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY_JSON="$REPO_ROOT/run_hash_summary_v50.json"

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git_hash="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo "nogit")"
py_ver="$(python -c 'import platform;print(platform.python_version())' 2>/dev/null || echo "python?")"

if command -v shasum >/dev/null 2>&1; then
  conf_hash="$(shasum -a 256 "$REPO_ROOT/$CONFIG_PATH" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  conf_hash="$(sha256sum "$REPO_ROOT/$CONFIG_PATH" | awk '{print $1}')"
else
  conf_hash="nohashbin"
fi

mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/run_hash.json" <<JSON
{
  "timestamp_utc": "$timestamp",
  "git_hash": "$git_hash",
  "python": "$py_ver",
  "config": "$CONFIG_PATH",
  "config_sha256": "$conf_hash"
}
JSON

# append/merge simple log into repo summary (line-delimited JSON for simplicity)
mkdir -p "$(dirname "$SUMMARY_JSON")"
echo "{\"timestamp_utc\":\"$timestamp\",\"git_hash\":\"$git_hash\",\"config\":\"$CONFIG_PATH\",\"config_sha256\":\"$conf_hash\",\"out_dir\":\"${OUT_DIR#"$REPO_ROOT/"}\"}" >> "$SUMMARY_JSON"

echo "[hash_config] Wrote: $OUT_DIR/run_hash.json and appended to $SUMMARY_JSON"
