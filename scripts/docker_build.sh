#!/usr/bin/env bash

# ==========================================================================================
# SpectraMind V50 - Docker Image Build
#
# Builds a CUDA-compatible image pinned for reproducible runs and Kaggle compatibility.
#
# Usage:
#   bash scripts/docker_build.sh [--tag spectramind:v50] [--file Dockerfile]
# ==========================================================================================

set -Eeuo pipefail
TAG="spectramind:v50"
DOCKERFILE="Dockerfile"

print_help() {
  cat <<'USAGE'
SpectraMind V50: docker_build.sh

USAGE:
  bash scripts/docker_build.sh [options]

OPTIONS:
  --tag NAME       Docker tag (default: spectramind:v50)
  --file PATH      Dockerfile path (default: Dockerfile)
  --help           Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --file) DOCKERFILE="$2"; shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 2 ;;
  esac
done

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "[docker_build] Missing Dockerfile: $DOCKERFILE"
  exit 3
fi

docker build -f "$DOCKERFILE" -t "$TAG" .
echo "[docker_build] Built image: $TAG"
