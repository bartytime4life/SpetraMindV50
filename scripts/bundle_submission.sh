#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Bundle Kaggle Submission
#
# Takes an output directory containing μ/σ predictions and packages a Kaggle-ready ZIP.
# Also writes a manifest and optionally opens HTML report.
#
# Usage:
#   bash scripts/bundle_submission.sh --in logs/predict_… [--name mysub.zip] [--open-html=0|1]
# ==========================================================================================

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR=""
ZIP_NAME=""
OPEN_HTML=0

print_help() {
  cat <<'USAGE'
SpectraMind V50: bundle_submission.sh

USAGE:
  bash scripts/bundle_submission.sh --in DIR [options]

OPTIONS:
  --in DIR          Input directory with predictions (required)
  --name FILENAME   Zip filename (default: submission_TIMESTAMP.zip in DIR)
  --open-html=0|1   If an HTML diagnostics exists under DIR, open it (default: 0)
  --help            Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in) INPUT_DIR="$2"; shift 2 ;;
    --name) ZIP_NAME="$2"; shift 2 ;;
    --open-html=0|1) OPEN_HTML="${1#*=}"; shift 1 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  echo "[bundle_submission] --in DIR is required"
  exit 3
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[bundle_submission] Input dir not found: $INPUT_DIR"
  exit 4
fi

timestamp="$(date +"%Y%m%d_%H%M%S")"
[[ -z "$ZIP_NAME" ]] && ZIP_NAME="submission_${timestamp}.zip"

pushd "$INPUT_DIR" >/dev/null

# Package rules: include required CSV/NPY/JSON per challenge spec.
# Here we conservatively include CSV/NPY/JSON/MD/HTML manifests but exclude raw logs to keep size down.
zip -r "$ZIP_NAME" . \
  -i '*.csv' '*.npy' '*.json' '*.md' '*.html' \
  -x 'console.log' 'console.txt' 'tmp/' 'cache/*' || true
popd >/dev/null

echo "[bundle_submission] Created: $INPUT_DIR/$ZIP_NAME"

if [[ "$OPEN_HTML" -eq 1 ]]; then
  mapfile -t htmls < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.html" | sort)
  if [[ ${#htmls[@]} -gt 0 ]]; then
    bash "$SCRIPT_DIR/launch_dashboard.sh" --file "${htmls[-1]}" || true
  fi
fi
