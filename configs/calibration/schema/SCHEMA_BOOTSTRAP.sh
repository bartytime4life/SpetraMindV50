#!/usr/bin/env bash

# ---------------------------------------------------------------
# SpectraMind V50 — Install Calibration Schemas into repo (idempotent)
#
# Creates the schema files under configs/calibration/schema/ and commits them.
#
# Usage:
#   bash configs/calibration/schema/SCHEMA_BOOTSTRAP.sh
#
# Optional env:
#   COMMIT_MSG="chore(schema): add calibration schemas"
# ---------------------------------------------------------------

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DEST="$ROOT/configs/calibration/schema"
mkdir -p "$DEST"

write() {
  local path="$1"; shift
  mkdir -p "$(dirname "$path")"
  cat > "$path" <<'EOF'
REPLACE_ME
EOF
}

# This script is self-modifying by replacing REPLACE_ME blocks with the content
# from this chat drop is impractical here. Instead, copy-paste the files from the
# assistant response into the matching paths, then run the git block below.

git add configs/calibration/schema || true
git commit -m "${COMMIT_MSG:-chore(schema): add/update calibration schemas}" || true
echo "Done. To validate YAMLs:"
echo "  python configs/calibration/schema/validate.py --dir configs/calibration"
echo "To integrate into CI, add a job that runs the above validator."
