#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Launch Diagnostics Dashboard (HTML)
#
# Opens the latest or specified diagnostics HTML in the default system browser.
#
# Usage:
#   bash scripts/launch_dashboard.sh [--file path/to/report.html] [--latest]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_help() {
  cat <<'USAGE'
SpectraMind V50: launch_dashboard.sh

USAGE:
  bash scripts/launch_dashboard.sh [--file path/to/report.html] [--latest]

OPTIONS:
  --file PATH   Specific diagnostics HTML file to open
  --latest      Find and open the most recent diagnostics HTML in logs/
  --help        Show help
USAGE
}

TARGET=""
LATEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) TARGET="$2"; shift 2 ;;
    --latest) LATEST=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

if [[ $LATEST -eq 1 ]]; then
  mapfile -t files < <(find "$REPO_ROOT/logs" -type f -name "*.html" -printf "%T@ %p\n" 2>/dev/null | sort -nr | awk '{print $2}')
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[launch_dashboard] No HTML files found in logs/"
    exit 4
  fi
  TARGET="${files[0]}"
fi

if [[ -z "$TARGET" ]]; then
  echo "[launch_dashboard] --file is required unless --latest is provided"
  exit 3
fi
if [[ ! -f "$TARGET" ]]; then
  echo "[launch_dashboard] File not found: $TARGET"
  exit 4
fi

open_cmd() {
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$1"
  elif command -v open >/dev/null 2>&1; then open "$1"
  elif command -v start >/dev/null 2>&1; then start "$1"
  else echo "Open $1 manually."; fi
}

echo "[launch_dashboard] Opening: $TARGET"
open_cmd "$TARGET" || true
