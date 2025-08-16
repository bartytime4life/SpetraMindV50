#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Analyze CLI Log
#
# Summarizes v50_debug_log.md into CSV/MD and optionally cleans duplicates.
#
# Usage:
#   bash scripts/analyze_log.sh [--csv out.csv] [--md out.md] [--clean]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$REPO_ROOT/v50_debug_log.md"
CSV_OUT=""
MD_OUT=""
CLEAN=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: analyze_log.sh

USAGE:
  bash scripts/analyze_log.sh [options]

OPTIONS:
  --csv PATH   Export a CSV table of parsed CLI entries
  --md PATH    Export Markdown table/summary
  --clean      Deduplicate entries before parsing (in-place)
  --help       Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV_OUT="$2"; shift 2 ;;
    --md) MD_OUT="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

if [[ ! -f "$LOG_FILE" ]]; then
  echo "[analyze_log] No v50_debug_log.md found."
  exit 3
fi

if [[ $CLEAN -eq 1 ]]; then
  tmp="$(mktemp)"
  awk '!seen[$0]++' "$LOG_FILE" > "$tmp" && mv "$tmp" "$LOG_FILE"
  echo "[analyze_log] Deduped $LOG_FILE"
fi

# Very lightweight parser: pull timestamp, command, config hash if present
TMP_CSV="$(mktemp)"
echo "timestamp,command,config_hash,out_dir" > "$TMP_CSV"
grep -E '^.' "$LOG_FILE" | while IFS= read -r line; do
  ts="$(echo "$line" | sed -n 's/^\[\([^]]*\)\].*/\1/p' | head -n1)"
  cmd="$(echo "$line" | sed -n 's/.*CMD:\s\([^ ]\+.*\)$/\1/p' | head -n1)"
  hash="$(echo "$line" | sed -n 's/.*HASH:\s\([0-9a-f]\{32,64\}\).*/\1/p' | head -n1)"
  outd="$(echo "$line" | sed -n 's/.*OUT:\s\([^ ]\+.*\)$/\1/p' | head -n1)"
  echo "$ts,$cmd,$hash,$outd" >> "$TMP_CSV"
done

if [[ -n "$CSV_OUT" ]]; then
  mv "$TMP_CSV" "$CSV_OUT"
  echo "[analyze_log] Wrote CSV -> $CSV_OUT"
else
  cat "$TMP_CSV"
  rm -f "$TMP_CSV"
fi

if [[ -n "$MD_OUT" ]]; then
  {
    echo "| timestamp | command | config_hash | out_dir |"
    echo "|---|---|---|---|"
    if [[ -n "$CSV_OUT" ]]; then
      tail -n +2 "$CSV_OUT" | while IFS=, read -r a b c d; do
        echo "| ${a//"/} | ${b//"/} | ${c//"/} | ${d//"/} |"
      done
    else
      echo "(CSV not persisted; re-run with --csv to generate MD table.)"
    fi
  } > "$MD_OUT"
  echo "[analyze_log] Wrote Markdown -> $MD_OUT"
fi
