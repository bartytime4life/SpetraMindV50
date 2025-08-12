#!/usr/bin/env bash

# SpectraMind V50 — Add Project Documents to Repo Documentation (Codex-friendly one-command tool)

# This script is idempotent and safe to re-run. It:
# 1) Creates a MkDocs docs site (if missing) with Material theme and a clear nav.
# 2) Imports all "project documents" from a source folder into docs/, normalizes names,
#    and (optionally) converts PDFs/DOCX/TXT to Markdown using pandoc (if installed).
# 3) Generates/updates docs/index.md with a clean, clickable index of all project docs.
# 4) Updates/creates mkdocs.yml with a structured nav, including "Architecture" and "Seed Docs".
# 5) Commits the docs to git and pushes to origin (main by default).
# 6) If a "codex" CLI exists on PATH, prints a one-liner you can feed to it (or it can run this).

# Usage:
#   bash scripts/add_project_docs.sh

# Notes:
# - By default, it will look for source docs in:
#     ./docs_src              (preferred if you’ve already staged files there)
#     OR /mnt/data            (the uploaded/seed location provided by your environment)
#   Set DOCS_SRC to override:  DOCS_SRC=/path/to/your/files bash scripts/add_project_docs.sh
# - Requires: bash, git. Optional: pandoc (to convert .pdf/.docx into .md stubs or full text).
# - If pandoc is unavailable for PDFs, we will create a .md "link stub" to the original file.
# - Branch default is "main". Override with: GIT_BRANCH=dev bash scripts/add_project_docs.sh
# - This script is verbose by design, with explicit logging and failure modes appropriate for CI.

set -euo pipefail
umask 022

# Configurable variables
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GIT_BRANCH="${GIT_BRANCH:-main}"
DOCS_SRC="${DOCS_SRC:-}"
SITE_NAME="${SITE_NAME:-SpectraMind V50 Docs}"
THEME_NAME="${THEME_NAME:-material}"
CODENAME="${CODENAME:-spectramind-v50}"
AUTHOR_NAME="${AUTHOR_NAME:-SpectraMind Team}"
COMMIT_MSG="${COMMIT_MSG:-docs: import project seed documents and update site nav}"
MKDOCS_FILE="${REPO_ROOT}/mkdocs.yml"
DOCS_DIR="${REPO_ROOT}/docs"
SEED_DIR="${DOCS_DIR}/seed"
PAPERS_DIR="${DOCS_DIR}/papers"
ATTACH_DIR="${DOCS_DIR}/assets"
INDEX_FILE="${DOCS_DIR}/index.md"
ARCH_MD_SOURCE_CANDIDATES=(
  "${REPO_ROOT}/ARCHITECTURE.md"
  "${REPO_ROOT}/docs/ARCHITECTURE.md"
  "/mnt/data/ARCHITECTURE.md"
)

log() { printf "[add-docs] %s\n" "$*" >&2; }

die() { printf "[add-docs:ERROR] %s\n" "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

git_check_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository."
  if ! git remote -v | grep -qE 'origin\s+.+push'; then
    die "No git remote 'origin' configured. Add one: git remote add origin <url>"
  fi
}

choose_docs_src() {
  if [[ -n "${DOCS_SRC}" ]]; then
    [[ -d "${DOCS_SRC}" ]] || die "DOCS_SRC set but not a directory: ${DOCS_SRC}"
    printf "%s" "${DOCS_SRC}"
    return
  fi
  if [[ -d "${REPO_ROOT}/docs_src" ]]; then
    printf "%s" "${REPO_ROOT}/docs_src"
    return
  fi
  if [[ -d "/mnt/data" ]]; then
    printf "%s" "/mnt/data"
    return
  fi
  die "No source docs directory found. Create ./docs_src or set DOCS_SRC=/path/to/files"
}

slugify() {
  # Lowercase; replace non-alnum with hyphens; collapse duplicates; trim hyphens
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/-+/-/g; s/^-//; s/-$//'
}

ensure_dirs() {
  mkdir -p "${DOCS_DIR}" "${SEED_DIR}" "${PAPERS_DIR}" "${ATTACH_DIR}"
}

ensure_mkdocs_yml() {
  if [[ ! -f "${MKDOCS_FILE}" ]]; then
    log "Creating mkdocs.yml"
    cat > "${MKDOCS_FILE}" <<EOF
site_name: ${SITE_NAME}
site_author: ${AUTHOR_NAME}
theme:
  name: ${THEME_NAME}
  features:
    - navigation.expand
    - navigation.tracking
    - content.code.copy
    - toc.integrate
markdown_extensions:
  - toc:
      permalink: true
  - admonition
  - codehilite
  - pymdownx.details
  - pymdownx.superfences
nav:
  - Home: index.md
  - Architecture: ARCHITECTURE.md
  - Seed Docs:
      - Overview: seed/README.md
      # Seed documents will be appended here by this script.
  - Papers (Converted):
      - Overview: papers/README.md
      # Converted papers will be appended here by this script.
EOF
  else
    log "mkdocs.yml exists; will update nav entries if needed."
  fi
}

append_nav_entry() {
  local nav_path="$1"
  local yaml_path_rel="$2"

  # If entry already present, skip
  if grep -q -F "  - ${nav_path}: ${yaml_path_rel}" "${MKDOCS_FILE}"; then
    return 0
  fi

  # Append under appropriate section. If sections missing, ensure minimal structure exists.
  if ! grep -q -F "  - Seed Docs:" "${MKDOCS_FILE}"; then
    log "Adding 'Seed Docs' section to mkdocs.yml"
    awk '{print} /- Architecture:/ && !x {print "  - Seed Docs:\n      - Overview: seed/README.md"; x=1}' "${MKDOCS_FILE}" > "${MKDOCS_FILE}.tmp"
    mv "${MKDOCS_FILE}.tmp" "${MKDOCS_FILE}"
  fi
  if ! grep -q -F "  - Papers (Converted):" "${MKDOCS_FILE}"; then
    log "Adding 'Papers (Converted)' section to mkdocs.yml"
    awk '{print} /- Seed Docs:/ && !x {print "  - Papers (Converted):\n      - Overview: papers/README.md"; x=1}' "${MKDOCS_FILE}" > "${MKDOCS_FILE}.tmp"
    mv "${MKDOCS_FILE}.tmp" "${MKDOCS_FILE}"
  fi

  # Decide which block to append to by prefix of yaml_path_rel
  if [[ "${yaml_path_rel}" == seed/* ]]; then
    awk -v entry="      - ${nav_path}: ${yaml_path_rel}" '
BEGIN{added=0}
{print}
/^  - Seed Docs:$/ && !added {print entry; added=1}
' "${MKDOCS_FILE}" > "${MKDOCS_FILE}.tmp"
    mv "${MKDOCS_FILE}.tmp" "${MKDOCS_FILE}"
  elif [[ "${yaml_path_rel}" == papers/* ]]; then
    awk -v entry="      - ${nav_path}: ${yaml_path_rel}" '
BEGIN{added=0}
{print}
/^  - Papers \(Converted\):$/ && !added {print entry; added=1}
' "${MKDOCS_FILE}" > "${MKDOCS_FILE}.tmp"
    mv "${MKDOCS_FILE}.tmp" "${MKDOCS_FILE}"
  else
    # Top-level fallback
    echo "  - ${nav_path}: ${yaml_path_rel}" >> "${MKDOCS_FILE}"
  fi
}

ensure_seed_overviews() {
  [[ -f "${SEED_DIR}/README.md" ]] || cat > "${SEED_DIR}/README.md" <<'EOF'
# Seed Documents Overview

This section contains the source project documents (PDF, DOCX, TXT) used to guide the architecture, engineering, and scientific approach of SpectraMind V50. Where possible, these are converted to Markdown and cross-linked into the site navigation. If an original document remains binary-only, we include a Markdown "link stub" pointing to the file inside docs/assets/.
EOF

  [[ -f "${PAPERS_DIR}/README.md" ]] || cat > "${PAPERS_DIR}/README.md" <<'EOF'
# Papers (Converted)

This section contains Markdown conversions of selected PDFs/DOCX/TXT for easier reading and cross-linking within the docs site. Conversions are best-effort (via pandoc). Always refer to the original files under docs/assets/ for authoritative formatting.
EOF
}

ensure_index() {
  if [[ ! -f "${INDEX_FILE}" ]]; then
    log "Creating docs/index.md"
    cat > "${INDEX_FILE}" <<'EOF'
# SpectraMind V50 Documentation

Welcome to the SpectraMind V50 documentation site.

- Architecture: Core mission architecture & engineering doctrine.
- Seed Docs: Source project materials (PDF/DOCX/TXT).
- Papers (Converted): Best-effort Markdown conversions for convenient reading.

Use the left navigation to explore. This site is generated by MkDocs with the Material theme. To build locally:

```bash
pipx run mkdocs serve

or

poetry run mkdocs serve

or

mkdocs serve
```
EOF
  fi
}

detect_pandoc() {
  if command -v pandoc >/dev/null 2>&1; then
    echo "yes"
  else
    echo "no"
  fi
}

copy_asset() {
  local src="$1"
  local bn="$(basename "$src")"
  local slug="$(slugify "${bn%.*}")"
  local ext="${bn##*.}"
  local out="${ATTACH_DIR}/${slug}.${ext}"
  cp -f "$src" "$out"
  echo "$out"
}

make_link_stub_md() {
  local asset_path="$1"
  local title="$2"
  local out_md_path="$3"
  local rel_path="${asset_path#${DOCS_DIR}/}"
  cat > "${out_md_path}" <<EOF
# ${title}

This is a link stub for a binary document. Click the link below to open the original file.

- Original file: [${title}](../${rel_path})
EOF
}

convert_to_md_or_stub() {
  local src="$1"
  local title="$2"
  local out_dir="$3"      # where to write the markdown (seed/ or papers/)
  local pandoc_ok="$4"    # "yes" or "no"

  mkdir -p "${out_dir}"
  local slug="$(slugify "${title}")"
  local out_md="${out_dir}/${slug}.md"
  local asset_path

  asset_path="$(copy_asset "$src")"

  if [[ "${pandoc_ok}" == "yes" ]]; then
    # Try to convert; if conversion fails (e.g., encrypted PDF), fall back to stub.
    if pandoc "$src" -o "${out_md}" --from=auto --to=gfm -s >/dev/null 2>&1; then
      # Put a front header and a link to the original
      local rel_asset="${asset_path#${DOCS_DIR}/}"
      sed -i "1s;^;# ${title}\n\n> Converted with pandoc (best-effort). Original: [${title}](../${rel_asset})\n\n;" "${out_md}"
      echo "${out_md}"
      return
    fi
  fi

  # If no pandoc or conversion failed, create a stub that links to the original asset
  make_link_stub_md "${asset_path}" "${title}" "${out_md}"
  echo "${out_md}"
}

maybe_copy_architecture_md() {
  local found=""
  for c in "${ARCH_MD_SOURCE_CANDIDATES[@]}"; do
    if [[ -f "$c" ]]; then found="$c"; break; fi
  done
  if [[ -n "${found}" ]]; then
    cp -f "${found}" "${DOCS_DIR}/ARCHITECTURE.md"
    append_nav_entry "Architecture" "ARCHITECTURE.md"
  else
    log "ARCHITECTURE.md not found in candidates; skipping copy."
  fi
}

import_all_sources() {
  local src_root="$1"
  local pandoc_ok="$2"

  # Enumerate likely project documents (PDF, DOCX, TXT, MD)
  # We COPY/CONVERT from src_root to docs/ locations:
  # - PDF/DOCX/TXT -> Markdown in papers/ (converted) + asset copy in assets/
  # - Existing Markdown -> seed/ (as-is copy)

  local -a files
  IFS=$'\n' read -r -d '' -a files < <(find "${src_root}" -maxdepth 2 -type f \( -iname "*.pdf" -o -iname "*.docx" -o -iname "*.txt" -o -iname "*.md" \) -print0 | xargs -0 -I{} echo "{}" && printf '\0')

  if [[ "${#files[@]}" -eq 0 ]]; then
    log "No project documents found in ${src_root}"
    return
  fi

  # Prepare catalog lines for index
  local catalog_lines=()

  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    local base="$(basename "$f")"
    local title="${base%.*}"

    case "${f,,}" in
      *.md)
        # Copy markdown as-is into seed/
        local slug="$(slugify "${title}")"
        local dst_md="${SEED_DIR}/${slug}.md"
        cp -f "$f" "$dst_md"
        append_nav_entry "${title}" "seed/${slug}.md"
        catalog_lines+=("- [${title}](seed/${slug}.md)")
        ;;
      *.pdf|*.docx|*.txt)
        # Convert to MD (papers/) if possible; always copy asset
        local md_path
        md_path="$(convert_to_md_or_stub "${f}" "${title}" "${PAPERS_DIR}" "${pandoc_ok}")"
        local rel="${md_path#${DOCS_DIR}/}"
        append_nav_entry "${title}" "${rel}"
        if [[ "${rel}" == papers/* ]]; then
          catalog_lines+=("- [${title}](${rel})")
        else
          # Shouldn't happen, but guard anyway
          catalog_lines+=("- ${title}")
        fi
        ;;
      *)
        # skip
        ;;
    esac

  done

  # Update overview pages with a short catalog list (no long prose)
  if [[ "${#catalog_lines[@]}" -gt 0 ]]; then
    # Seed overview keeps its own simple list (existing content preserved)
    if ! grep -q "## Catalog" "${SEED_DIR}/README.md"; then
      {
        echo
        echo "## Catalog"
        echo
        for L in "${catalog_lines[@]}"; do
          echo "${L}"
        done
        echo
      } >> "${SEED_DIR}/README.md"
    fi
    if ! grep -q "## Catalog" "${PAPERS_DIR}/README.md"; then
      {
        echo
        echo "## Catalog"
        echo
        for L in "${catalog_lines[@]}"; do
          echo "${L}"
        done
        echo
      } >> "${PAPERS_DIR}/README.md"
    fi
  fi
}

git_commit_and_push() {
  # Ensure branch exists locally
  git rev-parse --abbrev-ref HEAD >/dev/null 2>&1 || die "Cannot determine current branch."
  local cur_branch
  cur_branch="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "${cur_branch}" != "${GIT_BRANCH}" ]]; then
    log "Current branch is '${cur_branch}', target is '${GIT_BRANCH}'. Keeping current branch."
  fi

  git add -A
  if git diff --cached --quiet; then
    log "No changes to commit."
  else
    git commit -m "${COMMIT_MSG}"
  fi

  # Push
  git push -u origin "${cur_branch}"
}

maybe_print_codex_hint() {
  if command -v codex >/dev/null 2>&1; then
    cat <<'EOF'

[codex hint]
You can run this same operation via Codex by invoking a shell tool. Example:

codex run --label "docs:import" -- '
  bash scripts/add_project_docs.sh
'

EOF
  fi
}

main() {
  require_cmd git
  git_check_repo

  ensure_dirs
  ensure_mkdocs_yml
  ensure_seed_overviews
  ensure_index

  # If there's a root ARCHITECTURE.md (or one in /mnt/data), copy it into docs/
  maybe_copy_architecture_md

  local src_root
  src_root="$(choose_docs_src)"
  log "Importing from source: ${src_root}"

  local have_pandoc
  have_pandoc="$(detect_pandoc)"
  if [[ "${have_pandoc}" == "yes" ]]; then
    log "pandoc detected — will convert PDFs/DOCX/TXT to Markdown where possible."
  else
    log "pandoc not found — will create Markdown link stubs to original binaries."
  fi

  import_all_sources "${src_root}" "${have_pandoc}"

  # Make sure mkdocs has a basic Architecture entry even if no ARCHITECTURE.md was found
  if ! grep -q -F "  - Architecture: ARCHITECTURE.md" "${MKDOCS_FILE}"; then
    if [[ -f "${DOCS_DIR}/ARCHITECTURE.md" ]]; then
      append_nav_entry "Architecture" "ARCHITECTURE.md"
    fi
  fi

  # Commit & push
  git_commit_and_push

  # Final notes (non-fatal)
  if command -v mkdocs >/dev/null 2>&1; then
    log "Build the site locally with: mkdocs serve"
  else
    log "MkDocs not found. Install with: pipx install mkdocs-material  (or use Poetry)"
  fi

  maybe_print_codex_hint
  log "All done."
}

main "$@"
