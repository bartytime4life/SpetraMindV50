#!/usr/bin/env bash
set -euo pipefail
DASHBOARD_FILE="diagnostic_report_latest.html"
if [[ -f "$DASHBOARD_FILE" ]]; then
    echo "[DASHBOARD] Opening $DASHBOARD_FILE..."
    xdg-open "$DASHBOARD_FILE" 2>/dev/null || open "$DASHBOARD_FILE"
else
    echo "[DASHBOARD] Dashboard file not found. Generate it first with bin/run-diagnose.sh"
fi
