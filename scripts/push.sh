#!/usr/bin/env bash
# SpectraMindV50 Smart Push — repo scan + format + test + version bump + changelog + sign+tag+push
# Usage:
#   scripts/push.sh [-m "message"] [-b branch] [-r remote] [--type auto|patch|minor|major]
#                   [--dry-run] [--no-tests] [--no-format] [--skip-stash]
#                   [--allow-dirty] [--no-tag] [--no-sign]
#
# Examples:
#   scripts/push.sh -m "cli: improve subcommands and UX" --type auto
#   scripts/push.sh --dry-run
#
set -Eeuo pipefail

# -------------- defaults --------------
MSG="${MSG:-}"                  # commit message (if empty, auto-generated)
BRANCH="${BRANCH:-}"            # default: current
REMOTE="${REMOTE:-origin}"
BUMP_TYPE="${BUMP_TYPE:-auto}"  # auto|patch|minor|major|none
DRYRUN=0
RUN_TESTS=1
RUN_FORMAT=1
SKIP_STASH=0
ALLOW_DIRTY=0
CREATE_TAG=1
SIGN_COMMIT=1

# -------------- tiny arg parser --------------
while (( "$#" )); do
  case "$1" in
    -m|--message)    MSG="$2"; shift 2 ;;
    -b|--branch)     BRANCH="$2"; shift 2 ;;
    -r|--remote)     REMOTE="$2"; shift 2 ;;
    --type)          BUMP_TYPE="$2"; shift 2 ;;
    --dry-run)       DRYRUN=1; shift ;;
    --no-tests)      RUN_TESTS=0; shift ;;
    --no-format)     RUN_FORMAT=0; shift ;;
    --skip-stash)    SKIP_STASH=1; shift ;;
    --allow-dirty)   ALLOW_DIRTY=1; shift ;;
    --no-tag)        CREATE_TAG=0; shift ;;
    --no-sign)       SIGN_COMMIT=0; shift ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# -------------- helpers --------------
log()   { printf "[%s] %s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die()   { log "ERROR: $*"; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }
repo_root() { git rev-parse --show-toplevel 2>/dev/null; }

ROOT="$(repo_root || true)"
[[ -n "$ROOT" ]] || die "Not a git repository (no .git found)"
cd "$ROOT"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ -n "$BRANCH" ]] || BRANCH="$CURRENT_BRANCH"

mkdir -p .logs
ts="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE=".logs/push_${ts}.log"
JSONL_FILE=".logs/events.jsonl"

# console + rotating file logging
exec > >(tee -a "$LOG_FILE") 2>&1

event() {
  # JSONL event stream
  local typ="$1"; shift
  local payload="$*"
  printf '{"ts":"%s","type":"%s","payload":%s}\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$typ" "${payload:-null}" >> "$JSONL_FILE"
}

log "SpectraMind Smart Push starting @ $ROOT"
event "start" "{\"branch\":\"$BRANCH\",\"remote\":\"$REMOTE\",\"bump\":\"$BUMP_TYPE\",\"dry_run\":$DRYRUN}"

# -------------- environment capture --------------
{
  echo "Git: $(git --version)"
  echo "Python: $(python3 -V 2>/dev/null || true)"
  echo "Poetry: $(poetry --version 2>/dev/null || true)"
  echo "Pip: $(pip --version 2>/dev/null || true)"
  echo "Ruff: $(ruff --version 2>/dev/null || true)"
  echo "Black: $(black --version 2>/dev/null || true)"
  echo "Pytest: $(pytest --version 2>/dev/null || true)"
  echo "git-cliff: $(git-cliff --version 2>/dev/null || true)"
} | sed 's/^/[tool]/'
event "env" "{\"branch\":\"$BRANCH\",\"remote\":\"$REMOTE\"}"

# -------------- scan repository --------------
log "Scanning repository changes…"
git status --porcelain=v1
CHANGED=$(git status --porcelain=v1 | awk '{print $2}')
CLI_CHANGED=$(git status --porcelain=v1 | awk '$2 ~ /^src\/spectramind\/cli/ {print $2}')

event "scan" "$(jq -cn --argjson total "$(git status --porcelain=v1 | wc -l | awk '{print $1}')" \
  --argjson cli "$(printf "%s\n" "$CLI_CHANGED" | grep -c . || echo 0)" \
  --arg files "$(printf "%s" "$CHANGED" | jq -R -s -c 'split("\n")|map(select(length>0))')" \
  '{total_changed:$total, cli_changed:$cli, files:$files}')"

if [[ $ALLOW_DIRTY -eq 0 ]]; then
  # Stash any local (unstaged/untracked) if requested
  if [[ $SKIP_STASH -eq 0 ]] && [[ -n "$(git status --porcelain)" ]]; then
    log "Stashing working tree (uncommitted changes)…"
    git stash push -u -m "smart-push-$ts" || true
    STASHED=1
  else
    STASHED=0
  fi
else
  STASHED=0
fi

# -------------- re-apply staged changes if stashed --------------
if [[ ${STASHED:-0} -eq 1 ]]; then
  log "Re-applying stash (keeping only tracked index changes as needed)…"
  git stash pop || true
fi

# -------------- format + lint --------------
if [[ $RUN_FORMAT -eq 1 ]]; then
  if have ruff; then
    log "Running ruff format & fix…"
    ruff format || true
    ruff check --fix || true
  elif have black; then
    log "Running black…"
    black . || true
  else
    log "No formatter found (ruff/black). Skipping."
  fi
fi

# -------------- tests --------------
if [[ $RUN_TESTS -eq 1 ]]; then
  if have pytest; then
    log "Running pytest…"
    pytest -q || die "Tests failed"
  else
    log "pytest not found; skipping tests."
  fi
fi

# -------------- determine version + bump --------------
get_version() {
  # try Poetry first
  if [[ -f pyproject.toml ]] && grep -q '^\[tool.poetry\]' pyproject.toml 2>/dev/null && have poetry; then
    poetry version -s
    return
  fi
  # fallback: read __init__.__version__
  if [[ -f src/spectramind/__init__.py ]]; then
    python3 - "$@" <<'PY'
import re,sys,Pathlib
from pathlib import Path
p=Path("src/spectramind/__init__.py")
m=re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", p.read_text())
print(m.group(1) if m else "")
PY
    return
  fi
  echo ""
}

set_version() {
  local new="$1"
  if [[ -f pyproject.toml ]] && grep -q '^\[tool.poetry\]' pyproject.toml 2>/dev/null && have poetry; then
    poetry version "$new" >/dev/null
    echo "$new"
    return
  fi
  if [[ -f src/spectramind/__init__.py ]]; then
    python3 - "$new" <<'PY'
import re,sys
new=sys.argv[1]
p="src/spectramind/__init__.py"
s=open(p).read()
if "__version__" in s:
  s=re.sub(r"(__version__\s*=\s*['\"])([^'\"]+)(['\"])", rf"\1{new}\3", s, count=1)
else:
  s=s.rstrip()+"\n__version__ = '"+new+"'\n"
open(p,"w").write(s)
print(new)
PY
    return
  fi
  echo "$new"
}

bump_semver() {
  local cur="$1" part="$2"
  python3 - "$cur" "$part" <<'PY'
import sys
v=sys.argv[1].strip()
part=sys.argv[2]
import re
m=re.match(r'^(\d+)\.(\d+)\.(\d+)(.*)?$', v)
if not m: 
  print(v); sys.exit(0)
maj,minor,patch,rest = map(int,[m.group(1),m.group(2),m.group(3)])+[0]
if part=="major": maj+=1; minor=0; patch=0
elif part=="minor": minor+=1; patch=0
elif part=="patch": patch+=1
print(f"{maj}.{minor}.{patch}")
PY
}

CURRENT_VERSION="$(get_version || true)"
log "Current version: ${CURRENT_VERSION:-unknown}"

# Heuristic for auto bump: if CLI changed -> minor, else patch; if message contains "BREAKING" -> major
auto_part="patch"
if git status --porcelain=v1 | awk '$2 ~ /^src\/spectramind\/cli/ {found=1} END{exit(!found)}'; then
  auto_part="minor"
fi
if [[ -n "$MSG" ]] && echo "$MSG" | grep -qi "BREAKING"; then
  auto_part="major"
fi

BUMP_PART="$auto_part"
case "$BUMP_TYPE" in
  auto)  ;; # keep auto
  patch|minor|major|none) BUMP_PART="$BUMP_TYPE" ;;
  *) die "Unknown bump type: $BUMP_TYPE" ;;
esac

NEW_VERSION="$CURRENT_VERSION"
if [[ "$BUMP_PART" != "none" ]] && [[ -n "$CURRENT_VERSION" ]]; then
  NEW_VERSION="$(bump_semver "$CURRENT_VERSION" "$BUMP_PART")"
  set_version "$NEW_VERSION" >/dev/null
  log "Version bumped: $CURRENT_VERSION -> $NEW_VERSION (part=$BUMP_PART)"
  event "version_bump" "{\"from\":\"$CURRENT_VERSION\",\"to\":\"$NEW_VERSION\",\"part\":\"$BUMP_PART\"}"
else
  log "No version bump performed."
fi

# -------------- changelog (optional) --------------
if have git-cliff; then
  log "Updating CHANGELOG.md via git-cliff…"
  git-cliff -o CHANGELOG.md --unreleased || true
fi

# -------------- compose commit message --------------
if [[ -z "$MSG" ]]; then
  # auto message focuses on CLI scope if changed
  if [[ -n "$CLI_CHANGED" ]]; then
    MSG="cli: update CLI modules; repo scan, format, tests, $( [[ -n "$NEW_VERSION" ]] && echo "v$NEW_VERSION" )"
  else
    MSG="chore: repo maintenance; format/tests $( [[ -n "$NEW_VERSION" ]] && echo "v$NEW_VERSION" )"
  fi
fi

# -------------- add + commit --------------
log "Staging changes…"
git add -A

if [[ $DRYRUN -eq 1 ]]; then
  log "[DRY-RUN] Would commit with message:"
  printf "\n----\n%s\n----\n" "$MSG"
else
  if [[ $SIGN_COMMIT -eq 1 ]]; then
    git commit -S -m "$MSG" || die "Commit failed"
  else
    git commit -m "$MSG" || die "Commit failed"
  fi
fi
event "commit" "$(jq -cn --arg msg "$MSG" '{message:$msg}')"

# -------------- tag --------------
TAG=""
if [[ $CREATE_TAG -eq 1 ]] && [[ -n "$NEW_VERSION" ]] && [[ "$NEW_VERSION" != "$CURRENT_VERSION" ]]; then
  TAG="v${NEW_VERSION}"
  if [[ $DRYRUN -eq 1 ]]; then
    log "[DRY-RUN] Would create tag $TAG"
  else
    git tag -a "$TAG" -m "Release $TAG" || die "Tag failed"
  fi
  event "tag" "$(jq -cn --arg tag "$TAG" '{tag:$tag}')"
fi

# -------------- push --------------
log "Pushing to $REMOTE $BRANCH (follow tags)…"
if [[ $DRYRUN -eq 1 ]]; then
  log "[DRY-RUN] git push $REMOTE $BRANCH --follow-tags"
else
  git push "$REMOTE" "$BRANCH" --follow-tags || die "Push failed"
fi
event "push" "$(jq -cn --arg remote "$REMOTE" --arg branch "$BRANCH" '{remote:$remote,branch:$branch}')"

log "Done. Log: $LOG_FILE  |  Events: $JSONL_FILE"

