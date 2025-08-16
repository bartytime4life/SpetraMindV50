#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Log Cleanup & Rotation
#
# Deduplicates v50_debug_log.md entries, optionally trims old logs, and rotates console logs.
#
# Usage:
#   bash scripts/clean_logs.sh [--dedupe] [--keep N] [--rotate]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEDUPE=0
KEEP=0
ROTATE=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: clean_logs.sh

USAGE:
  bash scripts/clean_logs.sh [options]

OPTIONS:
  --dedupe       Deduplicate CLI entries in v50_debug_log.md
  --keep N       Keep only the N most recent directories under logs/
  --rotate       Gzip-compress *.log older than 7 days
  --help         Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dedupe) DEDUPE=1; shift ;;
    --keep) KEEP="$2"; shift 2 ;;
    --rotate) ROTATE=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

LOG_FILE="$REPO_ROOT/v50_debug_log.md"
if [[ $DEDUPE -eq 1 ]]; then
  if [[ -f "$LOG_FILE" ]]; then
    tmp="$(mktemp)"
    awk '!seen[$0]++' "$LOG_FILE" > "$tmp" && mv "$tmp" "$LOG_FILE"
    echo "[clean_logs] Deduped $LOG_FILE"
  else
    echo "[clean_logs] No v50_debug_log.md found. Skipping dedupe."
  fi
fi

if [[ "$KEEP" -gt 0 && -d "$REPO_ROOT/logs" ]]; then
  echo "[clean_logs] Keeping $KEEP most recent log directories."
  mapfile -t dirs < <(find "$REPO_ROOT/logs" -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | sort -nr | awk '{print $2}')
  count=0
  for d in "${dirs[@]}"; do
    ((count++))
    if [[ $count -le $KEEP ]]; then continue; fi
    rm -rf "$d" || true
  done
fi

if [[ $ROTATE -eq 1 && -d "$REPO_ROOT/logs" ]]; then
  find "$REPO_ROOT/logs" -type f -name "*.log" -mtime +7 -exec gzip -f {} \; || true
  echo "[clean_logs] Rotated old *.log files."
fi
