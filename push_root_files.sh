#!/usr/bin/env bash
set -euo pipefail
MSG="${1:-chore(root): scaffold/update root files}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

FILES=(
  "README.md"
  "LICENSE"
  "CHANGELOG.md"
  "CONTRIBUTING.md"
  "CODE_OF_CONDUCT.md"
  "SECURITY.md"
  "CITATION.cff"
  "VERSION"
  ".gitignore"
  ".dockerignore"
  ".gitattributes"
  ".editorconfig"
  ".pre-commit-config.yaml"
  "pyproject.toml"
  "mkdocs.yml"
  "Dockerfile"
  ".vscode/settings.json"
  "ARCHITECTURE.md"
)

git add "${FILES[@]}" 2>/dev/null || true
git commit -m "$MSG" || echo "Nothing to commit."
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push -u origin "$BRANCH"
echo "Pushed root files to $BRANCH."
