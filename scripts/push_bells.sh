#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# SpectraMind V50: Ultra "Bells & Whistles++" Multi-Directory Push
# - Auto-detects default branch and remote
# - Stages all key dirs (utils/reporting/symbolic/training/train/telemetry/calibration)
# - Adds configs, scripts, bin, tests, CI, devcontainer, vscode
# - Ensures .gitignore/.gitattributes (LFS) and outputs/.gitkeep
# - Optional: ruff, black, mypy, pytest (coverage+JUnit) if available
# - Optional: pre-commit, spectramind selftest, DVC push
# - Writes reproducibility entry to v50_debug_log.md
# - Creates signed/annotated tag and pushes tag
# Safe to re-run.
# ---------------------------------------------------------------------------

set -euo pipefail

has_cmd() { command -v "$1" >/dev/null 2>&1; }
add_if_exists() { for p in "$@"; do [ -e "$p" ] && git add "$p"; done; }
add_dir_if_exists() { for d in "$@"; do [ -d "$d" ] && git add "$d"; done; }

# --- repo sanity -----------------------------------------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "❌ Not a git repo. Run:  git init && git remote add origin <URL>  then re-run." >&2
  exit 1
fi

# Detect default branch
DEFAULT_BRANCH="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||' || true)"
if [ -z "${DEFAULT_BRANCH}" ]; then
  if git show-ref --verify --quiet refs/heads/main; then DEFAULT_BRANCH=main;
  elif git show-ref --verify --quiet refs/heads/master; then DEFAULT_BRANCH=master;
  else DEFAULT_BRANCH=main; fi
fi
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# --- housekeeping: ignores, outputs, artifacts ----------------------------
mkdir -p outputs logs artifacts .cache
touch outputs/.gitkeep

if [ ! -f .gitignore ]; then
  cat > .gitignore <<'GIGN'
# SpectraMind V50
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.build/
.dist/
.cache/
.env
.venv
venv/
logs/
artifacts/
outputs/*
!outputs/.gitkeep
# DVC
.dvc/tmp/
.dvc/cache/
# Notebooks
.ipynb_checkpoints/
# OS
.DS_Store
Thumbs.db
GIGN
else
  ensure_line() { grep -qxF "$1" .gitignore || echo "$1" >> .gitignore; }
  ensure_line "__pycache__/"
  ensure_line "*.pyc"
  ensure_line "*.pyo"
  ensure_line "*.pyd"
  ensure_line "*.egg-info/"
  ensure_line ".cache/"
  ensure_line ".env"
  ensure_line ".venv"
  ensure_line "venv/"
  ensure_line "logs/"
  ensure_line "artifacts/"
  ensure_line "outputs/*"
  ensure_line "!outputs/.gitkeep"
  ensure_line ".dvc/tmp/"
  ensure_line ".dvc/cache/"
  ensure_line ".ipynb_checkpoints/"
  ensure_line ".DS_Store"
  ensure_line "Thumbs.db"
fi

# Optional Git LFS for heavy assets
if has_cmd git && has_cmd git-lfs; then
  git lfs install --skip-repo 2>/dev/null || true
  if [ ! -f .gitattributes ]; then
    cat > .gitattributes <<'GATTR'
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.h5  filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.fits filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
GATTR
  fi
fi

# --- stage mission-ready dirs ---------------------------------------------
add_dir_if_exists \
  src/spectramind/utils \
  src/spectramind/reporting \
  src/spectramind/symbolic \
  src/spectramind/training \
  src/spectramind/train \
  src/spectramind/telemetry \
  src/spectramind/calibration

add_dir_if_exists \
  configs \
  scripts \
  bin \
  tests \
  .github \
  .devcontainer \
  .vscode

add_if_exists dvc.yaml dvc.lock .dvc .dvcignore outputs/.gitkeep .gitignore .gitattributes

# --- style/format pass (optional) -----------------------------------------
if has_cmd ruff; then
  echo "🔧 ruff fix..."
  ruff check --fix . || true
fi
if has_cmd black; then
  echo "🖤 black format..."
  black . || true
fi
git add -A

# --- pre-commit (optional) -------------------------------------------------
if has_cmd pre-commit && [ -f .pre-commit-config.yaml ]; then
  echo "⛓️  pre-commit..."
  pre-commit install >/dev/null 2>&1 || true
  pre-commit run --all-files || true
  git add -A
fi

# --- mypy (optional) -------------------------------------------------------
MYPY_STATUS=0
if has_cmd mypy; then
  echo "🧠 mypy (best-effort)..."
  mypy src || MYPY_STATUS=$?
  # Save report stub
  echo "mypy_exit_code=${MYPY_STATUS}" > artifacts/mypy_status.txt
  add_if_exists artifacts/mypy_status.txt
fi

# --- pytest smoke (optional) ----------------------------------------------
PYTEST_STATUS=0
if has_cmd pytest; then
  echo "🧪 pytest smoke with coverage (best-effort)..."
  # Prefer minimal smoke to keep it fast on mobile
  if has_cmd coverage; then
    coverage run -m pytest -q || PYTEST_STATUS=$?
    coverage xml -o artifacts/coverage.xml || true
    coverage html -d artifacts/coverage_html || true
  else
    pytest -q || PYTEST_STATUS=$?
  fi
  # JUnit if plugin available (pytest --junitxml works without plugin)
  pytest --maxfail=1 --disable-warnings -q --junitxml=artifacts/junit.xml || true
  add_dir_if_exists artifacts
fi

# --- reproducibility log ---------------------------------------------------
NOW_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
BUILD_HASH="$(git rev-parse --short HEAD || echo 'NA')"
LOG_FILE="v50_debug_log.md"
{
  echo "### Commit $(date +"%Y-%m-%d %H:%M:%S %Z")"
  echo "- timestamp_utc: ${NOW_ISO}"
  echo "- branch: ${CURRENT_BRANCH}"
  echo "- default_branch: ${DEFAULT_BRANCH}"
  echo "- staged_dirs: [utils, reporting, symbolic, training, train, telemetry, calibration, configs, scripts, bin, tests]"
  echo "- dvc_present: $( [ -f dvc.yaml ] && echo yes || echo no )"
  echo "- mypy_exit_code: ${MYPY_STATUS}"
  echo "- pytest_exit_code: ${PYTEST_STATUS}"
  echo "- commit_parent: ${BUILD_HASH}"
  echo "- cli_version: $(has_cmd spectramind && spectramind --version 2>/dev/null | head -n1 || echo 'NA')"
  echo
} >> "${LOG_FILE}"
git add "${LOG_FILE}"

# --- commit (GPG-signed if configured) ------------------------------------
COMMIT_MSG="SpectraMind V50: ultra bells-and-whistles push (dirs+configs+CI+DVC+format+mypy+pytest+log)"
if git config --get commit.gpgsign >/dev/null 2>&1; then
  git commit -S -m "$COMMIT_MSG" || true
else
  git commit -m "$COMMIT_MSG" || true
fi

# --- validate symbolic profiles (optional) ---------------------------------
if has_cmd python; then
python - <<'PYCODE' || true
try:
  from spectramind.symbolic.weights.profiles import list_profiles, validate_profile
  names = list_profiles()
  if names:
    bad=[]
    for n in names:
      try:
        validate_profile(n)
      except Exception as e:
        bad.append(f"{n}: {e}")
    if bad:
      print("❌ Symbolic profile validation failures:\n  - " + "\n  - ".join(bad))
    else:
      print("✅ Symbolic profiles OK:", ", ".join(names))
except Exception as e:
  print(f"(symbolic profile validation skipped) {e}")
PYCODE
fi

# --- push code -------------------------------------------------------------
if git config remote.origin.url >/dev/null 2>&1; then
  echo "🚀 Pushing to origin/${CURRENT_BRANCH}..."
  git push origin "${CURRENT_BRANCH}"
else
  echo "⚠️ No remote 'origin'. Add one: git remote add origin <URL>"
fi

# --- DVC push (optional) ---------------------------------------------------
if has_cmd dvc && [ -f dvc.yaml ]; then
  echo "📦 dvc push..."
  dvc push || true
fi

# --- spectramind deep selftest (optional) ----------------------------------
if has_cmd spectramind; then
  echo "🧪 spectramind test --deep (non-blocking)..."
  spectramind test --deep || true
fi

# --- CI hint ---------------------------------------------------------------
[ -d .github ] && echo "🧰 GitHub Actions workflows detected; CI will run after push."

# --- version tag -----------------------------------------------------------
SHORT_SHA="$(git rev-parse --short HEAD || echo 'unknown')"
TAG_NAME="v50-$(date -u +%Y%m%d-%H%M%S)-${SHORT_SHA}"
TAG_MSG="SpectraMind V50 release: ${TAG_NAME} (mypy=${MYPY_STATUS}, pytest=${PYTEST_STATUS})"
if git config --get tag.gpgsign >/dev/null 2>&1; then
  git tag -s "${TAG_NAME}" -m "${TAG_MSG}" || true
else
  git tag -a "${TAG_NAME}" -m "${TAG_MSG}" || true
fi
if git config remote.origin.url >/dev/null 2>&1; then
  echo "🏷️  Pushing tag ${TAG_NAME}..."
  git push origin "${TAG_NAME}" || true
fi

echo "✅ Done. Branch '${CURRENT_BRANCH}' updated, tag '${TAG_NAME}' pushed."
