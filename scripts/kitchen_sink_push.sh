#!/usr/bin/env bash
# Kitchen-Sink Push Protocol for SpectraMind V50
# - Formats (black, isort), lints (ruff), type-checks (mypy) if available
# - Runs pytest and deep selftest wiring (conf_helpers/tests included)
# - Conventional signed commit (fallback to unsigned)
# - Creates annotated tag
# - Pushes to current branch; if protected branch blocks, pushes a new branch and prints PR hint
# - Supports --force to proceed despite failures; --no-{format,lint,typecheck,tests,selftest}; --branch BR; --no-tag

set -Eeuo pipefail

# ------------------------------
# Defaults & CLI flags
# ------------------------------
DO_FORMAT=1
DO_LINT=1
DO_TYPECHECK=1
DO_TESTS=1
DO_SELFTEST=1
DO_TAG=1
FORCE=0
BRANCH_OVERRIDE=""
COMMIT_SCOPE="repo"
COMMIT_TYPE="chore"

usage() {
  cat <<'USAGE'
Kitchen Sink Push

Usage:
  bash scripts/kitchen_sink_push.sh [options]

Options:
  --force                 Continue even if checks fail
  --no-format             Skip code formatting (black, isort)
  --no-lint               Skip linting (ruff)
  --no-typecheck          Skip mypy
  --no-tests              Skip pytest
  --no-selftest           Skip deep selftest
  --no-tag                Skip creating an annotated tag
  --branch <name>         Push to this branch instead of the current branch
  --scope <scope>         Conventional commit scope (default: repo)
  --type <type>           Conventional commit type, e.g. feat|fix|chore (default: chore)
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --no-format) DO_FORMAT=0; shift ;;
    --no-lint) DO_LINT=0; shift ;;
    --no-typecheck) DO_TYPECHECK=0; shift ;;
    --no-tests) DO_TESTS=0; shift ;;
    --no-selftest) DO_SELFTEST=0; shift ;;
    --no-tag) DO_TAG=0; shift ;;
    --branch) BRANCH_OVERRIDE="${2:-}"; shift 2 ;;
    --scope) COMMIT_SCOPE="${2:-repo}"; shift 2 ;;
    --type) COMMIT_TYPE="${2:-chore}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ------------------------------
# Repo root + sanity checks
# ------------------------------
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found in PATH" >&2; exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "ERROR: Not inside a git repository." >&2; exit 2
fi
cd "$REPO_ROOT"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ -n "${BRANCH_OVERRIDE}" ]]; then
  TARGET_BRANCH="${BRANCH_OVERRIDE}"
else
  TARGET_BRANCH="${CURRENT_BRANCH}"
fi

echo "== Kitchen Sink Push =="
echo "Repo:        ${REPO_ROOT}"
echo "Branch:      ${TARGET_BRANCH} (current: ${CURRENT_BRANCH})"
echo "Flags:       format=${DO_FORMAT} lint=${DO_LINT} typecheck=${DO_TYPECHECK} tests=${DO_TESTS} selftest=${DO_SELFTEST} tag=${DO_TAG} force=${FORCE}"
echo

# Abort if nothing to commit
if [[ -z "$(git status --porcelain)" ]]; then
  echo "Nothing to commit. Working tree clean."
  exit 0
fi

echo "Pending changes:"
git status -s
echo

# ------------------------------
# Optional: pre-commit (if configured)
# ------------------------------
if [[ -f ".pre-commit-config.yaml" ]]; then
  if command -v pre-commit >/dev/null 2>&1; then
    echo "Running pre-commit hooks..."
    pre-commit install >/dev/null 2>&1 || true
    if ! pre-commit run --all-files; then
      if [[ "$FORCE" -eq 0 ]]; then
        echo "pre-commit failed. Re-run with --force to continue anyway." >&2
        exit 1
      else
        echo "pre-commit failed, continuing due to --force."
      fi
    fi
    echo
  fi
fi

# ------------------------------
# Formatters
# ------------------------------
if [[ "$DO_FORMAT" -eq 1 ]]; then
  if command -v black >/dev/null 2>&1; then
    echo "Running black..."
    if ! black .; then
      if [[ "$FORCE" -eq 0 ]]; then echo "black failed; aborting. Use --force to continue." >&2; exit 1; else echo "black failed, continuing (--force)"; fi
    fi
  fi
  if command -v isort >/dev/null 2>&1; then
    echo "Running isort..."
    if ! isort .; then
      if [[ "$FORCE" -eq 0 ]]; then echo "isort failed; aborting. Use --force to continue." >&2; exit 1; else echo "isort failed, continuing (--force)"; fi
    fi
  fi
  echo
fi

# ------------------------------
# Linters
# ------------------------------
if [[ "$DO_LINT" -eq 1 ]]; then
  if command -v ruff >/dev/null 2>&1; then
    echo "Running ruff (fix)..."
    if ! ruff check --fix .; then
      if [[ "$FORCE" -eq 0 ]]; then echo "ruff failed; aborting. Use --force to continue." >&2; exit 1; else echo "ruff failed, continuing (--force)"; fi
    fi
  fi
  echo
fi

# ------------------------------
# Type-check
# ------------------------------
if [[ "$DO_TYPECHECK" -eq 1 ]]; then
  if command -v mypy >/dev/null 2>&1; then
    echo "Running mypy..."
    # If config exists, mypy will pick it up; otherwise still runs
    if ! mypy .; then
      if [[ "$FORCE" -eq 0 ]]; then echo "mypy failed; aborting. Use --force to continue." >&2; exit 1; else echo "mypy failed, continuing (--force)"; fi
    fi
  fi
  echo
fi

# ------------------------------
# Tests
# ------------------------------
if [[ "$DO_TESTS" -eq 1 ]]; then
  if command -v pytest >/dev/null 2>&1; then
    echo "Running pytest (quiet)..."
    # Prefer a quick pass; CI can run full
    if ! pytest -q; then
      if [[ "$FORCE" -eq 0 ]]; then echo "pytest failed; aborting. Use --force to continue." >&2; exit 1; else echo "pytest failed, continuing (--force)"; fi
    fi
  else
    echo "pytest not found; skipping tests."
  fi
  echo
fi

# ------------------------------
# Deep selftest (wires conf_helpers/tests)
# ------------------------------
if [[ "$DO_SELFTEST" -eq 1 ]]; then
  if [[ -f "selftest.py" ]]; then
    echo "Running deep selftest..."
    if ! python selftest.py run --deep --export-md --export-json; then
      if [[ "$FORCE" -eq 0 ]]; then echo "selftest deep failed; aborting. Use --force to continue." >&2; exit 1; else echo "selftest deep failed, continuing (--force)"; fi
    fi
  else
    echo "selftest.py not found; skipping deep selftest."
  fi
  echo
fi

# ------------------------------
# Stage and build commit message
# ------------------------------
echo "Staging changes…"
git add -A

CONFIG_HASH="N/A"
if [[ -f "run_hash_summary_v50.json" ]]; then
  # Try jq first; fallback to python
  if command -v jq >/dev/null 2>&1; then
    CONFIG_HASH="$(jq -r '.config_hash // .hash // .run_hash // .config_hash_hex // "N/A"' run_hash_summary_v50.json)"
  else
    CONFIG_HASH="$(python - <<'PY'
import json,sys
try:
  d=json.load(open("run_hash_summary_v50.json"))
  for k in ("config_hash","hash","run_hash","config_hash_hex"):
    v=d.get(k)
    if isinstance(v,str) and v.strip():
      print(v); break
  else:
    print("N/A")
except Exception:
  print("N/A")
PY
)"
  fi
fi

DATE_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
MSG_HEADER="${COMMIT_TYPE}(${COMMIT_SCOPE}): kitchen-sink push — format/lint/tests/selftest wiring"
MSG_BODY=$'\n'"- ci:selftest: deep wired; conf_helpers/tests included"$'\n'"- config_hash: ${CONFIG_HASH}"$'\n'"- datetime_utc: ${DATE_UTC}"$'\n'"- branch: ${TARGET_BRANCH}"

echo
echo "Committing…"
if ! git commit -S -m "${MSG_HEADER}" -m "${MSG_BODY}"; then
  echo "Signed commit failed (no GPG key?) — committing unsigned."
  git commit -m "${MSG_HEADER}" -m "${MSG_BODY}" || { echo "git commit failed."; exit 1; }
fi

SHORTSHA="$(git rev-parse --short HEAD)"

# ------------------------------
# Tag
# ------------------------------
if [[ "$DO_TAG" -eq 1 ]]; then
  TAG="v50-kitchensink-$(date -u +%Y%m%d-%H%M)-${SHORTSHA}"
  echo "Tagging ${TAG}…"
  git tag -a "${TAG}" -m "Kitchen sink push @ ${DATE_UTC} | cfg=${CONFIG_HASH}"
fi

# ------------------------------
# Push (with protected-branch fallback)
# ------------------------------
REMOTE="origin"
echo "Pushing to ${REMOTE} ${TARGET_BRANCH}…"
if ! git push -u "${REMOTE}" "${TARGET_BRANCH}"; then
  echo "Direct push failed (protected/main?). Creating topic branch…"
  TOPIC="kitchen/$(date -u +%Y%m%d-%H%M)-${SHORTSHA}"
  git switch -c "${TOPIC}"
  git push -u "${REMOTE}" "${TOPIC}"
  echo
  echo "Pushed to ${TOPIC}."
  echo "If GitHub CLI is installed, open a PR with:"
  echo "  gh pr create --fill --base ${TARGET_BRANCH} --head ${TOPIC}"
else
  echo "Push succeeded."
fi

# Push tags (if any)
if [[ "$DO_TAG" -eq 1 ]]; then
  echo "Pushing tags…"
  git push --tags || { echo "WARN: tag push failed (non-fatal)"; }
fi

echo
echo "== Done =="
echo "Commit: $(git rev-parse HEAD)  (${SHORTSHA})"
echo "Branch: ${TARGET_BRANCH} (or topic branch if fallback used)"
echo "Config hash: ${CONFIG_HASH}"
