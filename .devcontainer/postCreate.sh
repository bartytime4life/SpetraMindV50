#!/usr/bin/env bash
set -euo pipefail

echo "[postCreate] Starting SpectraMind V50 setup..."

# Ensure we are at repo root inside the container
if [ ! -d ".git" ]; then
  echo "[postCreate] WARNING: .git not found at $(pwd). If this isn't repo root, cd to it."
fi

# Git safe directory (avoid "detected dubious ownership" in Codespaces/Bind mounts)
git config --global --add safe.directory "*"

# Python venv (project-local) if Poetry isn't used
create_local_venv() {
  echo "[postCreate] Creating local .venv and installing requirements (fallback)..."
  python -m venv .venv
  source .venv/bin/activate
  if [ -f "requirements.txt" ]; then
    pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt || true
  fi
}

# Prefer Poetry if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
  echo "[postCreate] pyproject.toml detected — using Poetry."
  export PATH="$HOME/.local/bin:$PATH"
  poetry --version || (echo "[postCreate] Poetry missing" && exit 1)
  # Configure poetry to create venv inside project
  poetry config virtualenvs.in-project true
  poetry env use python || true
  poetry install --no-root || true
  VENV_PATH="$(poetry env info --path 2>/dev/null || true)"
  if [ -n "${VENV_PATH}" ] && [ -d "${VENV_PATH}" ]; then
    echo "[postCreate] Poetry venv: ${VENV_PATH}"
  else
    echo "[postCreate] Poetry venv not detected; falling back to local venv."
    create_local_venv
  fi
else
  echo "[postCreate] No pyproject.toml; falling back to local venv."
  create_local_venv
fi

# Optional: pre-commit hooks if configured
if command -v poetry >/dev/null 2>&1 && poetry run pre-commit --version >/dev/null 2>&1; then
  echo "[postCreate] Installing pre-commit hooks (poetry)..."
  poetry run pre-commit install || true
elif [ -x ".venv/bin/pre-commit" ]; then
  echo "[postCreate] Installing pre-commit hooks (venv)..."
  .venv/bin/pre-commit install || true
fi

# Make sure typical dev dirs exist
mkdir -p data logs artifacts submissions reports

# Gentle checks — won't fail container create
echo "[postCreate] Running gentle repo checks..."
if [ -f "spectramind.py" ]; then
  (poetry run python spectramind.py --version 2>/dev/null || true) || ( ./.venv/bin/python spectramind.py --version 2>/dev/null || true )
fi
if [ -f "selftest.py" ]; then
  (poetry run python selftest.py --mode fast 2>/dev/null || true) || ( ./.venv/bin/python selftest.py --mode fast 2>/dev/null || true )
fi

echo "[postCreate] Done."
