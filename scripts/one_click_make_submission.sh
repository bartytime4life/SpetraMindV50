#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - One-Click Make Submission
#
# End-to-end flow: selftest -> predict -> bundle -> (optional) open HTML
#
# Usage:
#   bash scripts/one_click_make_submission.sh --ckpt ckpts/best.pt [--open-html]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT=""
OPEN_HTML=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: one_click_make_submission.sh

USAGE:
  bash scripts/one_click_make_submission.sh --ckpt PATH [--open-html]

OPTIONS:
  --ckpt PATH    Model checkpoint path (required)
  --open-html    Open diagnostics HTML after bundling
  --help         Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --open-html) OPEN_HTML=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "[one_click] --ckpt is required"
  exit 3
fi

bash "$SCRIPT_DIR/run_selftest.sh" --deep || { echo "[one_click] selftest failed"; exit 5; }
bash "$SCRIPT_DIR/predict_v50.sh" --ckpt "$CKPT" --bundle --open-html=$OPEN_HTML || { echo "[one_click] predict/bundle failed"; exit 6; }
echo "[one_click] Submission done."
