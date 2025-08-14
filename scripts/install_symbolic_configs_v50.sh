#!/usr/bin/env bash
set -euo pipefail
umask 022

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "[SpectraMind] Symbolic config pack is located at: $ROOT/configs/symbolic"
echo "-------------------------------------------------------------"
echo "Created/updated symbolic config tree under: $ROOT/configs/symbolic"
echo "Key entrypoint: configs/symbolic/base.yaml"
echo
echo "Quick compose examples:"
echo "  python -m spectramind train +config_name=symbolic/base.yaml symbolic.profile=leaderboard"
echo "  python -m spectramind diagnose dashboard symbolic.profile=diagnostics"
echo
echo "Git commands to track changes:"
echo "  git add configs/symbolic"
echo "  git commit -m \"feat(configs/symbolic): add/refresh symbolic configs\""
echo "  git push"
echo "-------------------------------------------------------------"
