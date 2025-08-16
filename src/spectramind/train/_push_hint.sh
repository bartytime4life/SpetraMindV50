#!/usr/bin/env bash

# SpectraMind V50 — Convenience Push Script (train package only)

# Usage: bash src/spectramind/train/_push_hint.sh "Update train package"

set -euo pipefail
MSG="${1:-Update SpectraMind V50 train package}"
git add src/spectramind/train
git commit -m "${MSG}"
git push

# End of file

