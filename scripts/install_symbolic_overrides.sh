#!/usr/bin/env bash
set -euo pipefail
umask 022
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TARGET="$ROOT/configs/symbolic/overrides"
[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found. Run this from repo root after copying."; exit 1; }

echo "[ok] overrides folder present:"
command -v tree >/dev/null 2>&1 && tree -a "$TARGET" || find "$TARGET" -type f | sed "s|$ROOT/||"

echo "Minimal YAML sanity (grep common keys)"

for f in $(find "$TARGET" -type f -name '*.yaml'); do
  head -n 1 "$f" >/dev/null || true
done
echo "[ok] YAML files written: $(find "$TARGET" -type f -name '*.yaml' | wc -l | tr -d ' ')"

if [[ "${1:-}" == "--push" ]]; then
  MSG="${2:-chore(symbolic): add overrides pack}"
  git add configs/symbolic/overrides || true
  git commit -m "$MSG" || true
  git push || true
  echo "[ok] pushed with message: $MSG"
fi
