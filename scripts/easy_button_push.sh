#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# scripts/easy_button_push.sh
# Master-Coder-Protocol — One-Click Bootstrap → Commit → Push → Enable Pages
# -----------------------------------------------------------------------------

# What it does:
# - Verifies you're in a git repo and that GitHub CLI is logged in
# - Ensures branch "main" exists and is the default
# - Runs the bootstrap script to generate all files
# - Commits and pushes everything to origin/main
# - Enables GitHub Pages for workflow (Actions) builds
# - Triggers the docs workflow so your site goes live
#
# Usage:
#   bash scripts/easy_button_push.sh
#
# Prereqs:
# - Git installed and repo cloned locally
# - GitHub CLI installed (`gh`) and authed: `gh auth login`
# -----------------------------------------------------------------------------
set -euo pipefail
umask 022

echo "🔧 Easy Button: starting…"

# --- repo root guard ----------------------------------------------------------
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "❌ Not inside a git repository. cd into your repo and try again."
  exit 1
fi
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# --- gh auth check ------------------------------------------------------------
if ! command -v gh >/dev/null 2>&1; then
  echo "❌ GitHub CLI (gh) not found. Install from https://cli.github.com/ and run: gh auth login"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "❌ You are not logged into GitHub CLI. Run: gh auth login"
  exit 1
fi

# --- resolve owner/repo -------------------------------------------------------
# Prefer gh json introspection (robust to https/ssh remotes)
if gh repo view --json name,owner >/dev/null 2>&1; then
  REPO_NAME="$(gh repo view --json name -q .name)"
  REPO_OWNER="$(gh repo view --json owner -q .owner.login)"
else
  # Fallback: parse from git remote
  REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -z "$REMOTE_URL" ]]; then
    echo "❌ No 'origin' remote configured. Add one, e.g.: git remote add origin git@github.com:YOURUSER/YOURREPO.git"
    exit 1
  fi
  # Handle SSH and HTTPS formats
  if [[ "$REMOTE_URL" =~ git@github.com:(.*)/(.*).git ]]; then
    REPO_OWNER="${BASH_REMATCH[1]}"
    REPO_NAME="${BASH_REMATCH[2]}"
  elif [[ "$REMOTE_URL" =~ https://github.com/(.*)/(.*)(.git)? ]]; then
    REPO_OWNER="${BASH_REMATCH[1]}"
    REPO_NAME="${BASH_REMATCH[2]}"
  else
    echo "❌ Could not parse origin remote URL: $REMOTE_URL"
    exit 1
  fi
fi
FULL_REPO="$REPO_OWNER/$REPO_NAME"
echo "📦 Repo detected: $FULL_REPO"

# --- ensure main branch exists and is default --------------------------------
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  if git show-ref --verify --quiet refs/heads/main; then
    echo "↪ Switching to existing 'main' branch…"
    git checkout main
  else
    echo "➕ Creating 'main' branch…"
    git checkout -b main
  fi
fi

# Set default branch to main (safe if already set)
echo "⚙️  Setting default branch to 'main' (if needed)…"
gh api -X PATCH "repos/$FULL_REPO" -f default_branch='main' >/dev/null || true

# --- run bootstrap ------------------------------------------------------------
BOOTSTRAP="scripts/bootstrap_master_coder_protocol.sh"
if [[ ! -x "$BOOTSTRAP" ]]; then
  if [[ -f "$BOOTSTRAP" ]]; then
    chmod +x "$BOOTSTRAP"
  else
    echo "❌ Missing $BOOTSTRAP."
    echo "   Make sure you pasted the bootstrap script we generated earlier at: $BOOTSTRAP"
    exit 1
  fi
fi

echo "🚀 Running bootstrap to materialize files…"
bash "$BOOTSTRAP"

# --- sanity checks (lightweight, non-fatal) ----------------------------------
[[ -f "README.md" ]] || echo "⚠️  README.md not found post-bootstrap (continuing)…"
[[ -f "mkdocs.yml" ]] || echo "⚠️  mkdocs.yml not found post-bootstrap (continuing)…"
[[ -f ".github/workflows/docs.yml" ]] || echo "⚠️  docs workflow missing (continuing)…"

# --- optional: local quick doc build to catch errors --------------------------
if command -v mkdocs >/dev/null 2>&1; then
  echo "🔍 Building docs locally (mkdocs build) to catch errors…"
  mkdocs build || { echo "❌ mkdocs build failed; fix docs warnings/errors and re-run."; exit 1; }
else
  echo "ℹ️  mkdocs not installed locally; skipping local build (CI will build)."
fi

# --- stage, commit, push ------------------------------------------------------
echo "📝 Staging changes…"
git add -A

# Only commit if there are staged changes
if ! git diff --cached --quiet; then
  echo "✅ Committing…"
  git commit -m "feat: bootstrap Master Coder Protocol (docs, examples, CI, trackers)"
else
  echo "ℹ️  No changes to commit (already up to date)."
fi

# Ensure remote exists
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "❌ No 'origin' remote set. Add it, e.g.:"
  echo "   git remote add origin git@github.com:$FULL_REPO.git"
  exit 1
fi

echo "⬆️  Pushing to origin/main…"
git push -u origin main

# --- enable GitHub Pages for workflow builds ---------------------------------
# This sets Pages "build_type=workflow", which is what actions/deploy-pages uses.
echo "🌐 Enabling GitHub Pages (workflow builds)…"
# Try create; if it exists, patch; ignore errors to be idempotent
gh api -X POST "repos/$FULL_REPO/pages" -f build_type='workflow' >/dev/null 2>&1 || \
  gh api -X PUT  "repos/$FULL_REPO/pages" -f build_type='workflow' >/dev/null 2>&1 || true

# --- trigger docs CI (docs.yml) ----------------------------------------------
# If the docs workflow exists, dispatch it; else, rely on the push that just happened.
if gh workflow list --limit 200 | grep -q "^docs"; then
  echo "🏁 Dispatching docs workflow…"
  gh workflow run docs.yml >/dev/null 2>&1 || true
else
  echo "ℹ️  No docs workflow registered via gh yet; your push should have already started CI."
fi

# --- report URLs --------------------------------------------------------------
REPO_HTML_URL="https://github.com/$FULL_REPO"
PAGES_URL="$(gh api "repos/$FULL_REPO/pages" -q '.html_url' 2>/dev/null || echo "")"

echo "--------------------------------------------------------------------"
echo "🎉 Done!"
echo "Repo:   $REPO_HTML_URL"
if [[ -n "$PAGES_URL" && "$PAGES_URL" != "null" ]]; then
  echo "Docs:   $PAGES_URL"
else
  echo "Docs:   (will appear after first successful docs deploy action)"
fi
echo "--------------------------------------------------------------------"
echo "Tip: create a Python venv and pre-commit hooks for local dev:"
echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pre-commit install"
echo "--------------------------------------------------------------------"

# End of script
