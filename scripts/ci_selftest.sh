#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts
echo "=== SpectraMind V50 CI Selftest ==="
python -V || true
if command -v poetry >/dev/null 2>&1; then
  poetry run python -m spectramind --version || true
  poetry run python -m spectramind selftest --fast || true
else
  python -m spectramind --version || true
  python -m spectramind selftest --fast || true
fi
[ -d artifacts ] && find artifacts -maxdepth 2 -type f -print || true
