#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# SpectraMind V50 — Configs Push (phone‑friendly, idempotent)
# - Creates/refreshes the five overrides/_schemas files (unless FORCE=1).
# - Stages and commits ALL of configs/ in one atomic push.
# - Safe on existing repos; auto-inits if needed.
# Usage:
#   REMOTE_URL="https://github.com/you/yourrepo.git" bash scripts/push-configs.sh
# Options (env):
#   BRANCH=main|master
#   FORCE=1          # overwrite existing schema files
#   COMMIT_MSG="..." # custom commit message
#   ADD_PATHS="path1 path2" # extra paths to stage in addition to configs/
# ------------------------------------------------------------------------------

set -euo pipefail
umask 022

REMOTE_URL="${REMOTE_URL:-}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-chore(configs): add/refresh symbolic override schemas + push all configs}"
STAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
FORCE="${FORCE:-0}"

# ---- helpers -----------------------------------------------------------------
write_file() {
  local path="$1"; local content="$2"
  if [[ -f "$path" && "$FORCE" != "1" ]]; then
    printf "SKIP (exists): %s\n" "$path"
    return 0
  fi
  mkdir -p "$(dirname "$path")"
  printf "WRITE: %s\n" "$path"
  printf "%s" "$content" > "$path"
}

need_git() {
  command -v git >/dev/null 2>&1 || { echo "ERROR: git not found"; exit 1; }
}

# ---- ensure schema files exist -----------------------------------------------
need_git

mkdir -p configs/symbolic/overrides/_schemas

write_file "configs/symbolic/overrides/_schemas/base.yaml" "$(cat <<"YAML"
# Schema: Base symbolic override structure
$schema: "http://json-schema.org/draft-07/schema#"
title: "Symbolic Override - Base Schema"
type: object
description: |
  Base schema for symbolic overrides, defining the minimal set of keys required
  to override default symbolic constraints, loss weights, or rule parameters.
properties:
  id:
    type: string
    description: Unique identifier for this override set.
  description:
    type: string
    description: Human-readable description of the override purpose/context.
  enabled:
    type: boolean
    default: true
    description: Whether this override set is active.
  rules:
    type: array
    description: List of rule overrides.
    items:
      type: object
      properties:
        name:
          type: string
          description: Name of the symbolic rule being overridden.
        weight:
          type: number
          description: New loss weight for this rule.
        params:
          type: object
          description: Optional rule-specific parameters to override.
      required: ["name"]
required: [id, rules]
additionalProperties: false
YAML
)"

write_file "configs/symbolic/overrides/_schemas/molecule_region.yaml" "$(cat <<"YAML"
# Schema: Molecule-region symbolic overrides
$schema: "http://json-schema.org/draft-07/schema#"
title: "Symbolic Override - Molecule Region Schema"
type: object
description: |
  Target symbolic overrides to specific molecule absorption regions or wavelength bands.
properties:
  id: { type: string, description: Unique identifier for this override set. }
  target_molecules:
    type: array
    description: Molecules whose spectral regions are targeted by these overrides.
    items: { type: string, enum: ["H2O", "CO2", "CH4", "NH3", "O3", "CO", "Other"] }
  wavelength_range:
    type: array
    description: Two-element array [start_um, end_um] in microns.
    items: { type: number }
    minItems: 2
    maxItems: 2
  rules:
    type: array
    items:
      type: object
      properties:
        name: { type: string, description: Name of the symbolic rule being overridden. }
        weight: { type: number, description: New loss weight for this rule. }
      required: ["name"]
required: [id, target_molecules, wavelength_range, rules]
additionalProperties: false
YAML
)"

write_file "configs/symbolic/overrides/_schemas/calibration.yaml" "$(cat <<"YAML"
# Schema: Symbolic overrides for calibration/training phases
$schema: "http://json-schema.org/draft-07/schema#"
title: "Symbolic Override - Calibration Schema"
type: object
description: |
  Apply override rules conditionally in phases (pretrain, contrastive, finetune, calibration, corel).
properties:
  id: { type: string }
  phase: { type: string, enum: ["pretrain", "contrastive", "finetune", "calibration", "corel"] }
  rules:
    type: array
    items:
      type: object
      properties:
        name: { type: string, description: Name of the symbolic rule being overridden. }
        weight: { type: number, description: New loss weight for this rule. }
        params: { type: object, description: Optional rule-specific parameter overrides. }
      required: ["name"]
required: [id, phase, rules]
additionalProperties: false
YAML
)"

write_file "configs/symbolic/overrides/_schemas/uncertainty.yaml" "$(cat <<"YAML"
# Schema: Symbolic overrides for uncertainty-aware training
$schema: "http://json-schema.org/draft-07/schema#"
title: "Symbolic Override - Uncertainty Schema"
type: object
description: |
  Modify symbolic loss components that interact with sigma (σ) predictions or calibration objectives.
properties:
  id: { type: string }
  applies_to_sigma: { type: boolean, default: true, description: Whether overrides touch σ-head constraints. }
  sigma_weight_multiplier: { type: number, default: 1.0, description: Global multiplier for σ-linked symbolic losses. }
  rules:
    type: array
    items:
      type: object
      properties:
        name: { type: string, description: Name of the symbolic rule being overridden. }
        weight: { type: number, description: New loss weight for this rule. }
      required: ["name"]
required: [id, rules]
additionalProperties: false
YAML
)"

write_file "configs/symbolic/overrides/_schemas/smoothness.yaml" "$(cat <<"YAML"
# Schema: Symbolic overrides for spectral smoothness constraints
$schema: "http://json-schema.org/draft-07/schema#"
title: "Symbolic Override - Smoothness Schema"
type: object
description: |
  Adjust smoothness loss terms globally or per-targeted bin ranges.
properties:
  id: { type: string }
  global_weight: { type: number, description: Default smoothness weight across all bins. }
  targeted_ranges:
    type: array
    description: Optional bin ranges with custom weights.
    items:
      type: object
      properties:
        start_bin: { type: integer }
        end_bin: { type: integer }
        weight: { type: number }
      required: ["start_bin", "end_bin", "weight"]
required: [id, global_weight]
additionalProperties: false
YAML
)"

# ---- minimal .gitignore (first-run convenience) ------------------------------
if [[ ! -f .gitignore ]]; then
  cat > .gitignore <<EOF
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
# Build/Artifacts
dist/
build/
# Data/Cache
.dvc/
.cache/
# Editors
.vscode/
.idea/
EOF
  echo "WRITE: .gitignore"
fi

# ---- git init/remote/branch --------------------------------------------------
if [[ ! -d .git ]]; then
  echo "INIT: git repository"
  git init -b "$BRANCH"
else
  git checkout -B "$BRANCH"
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "REMOTE: origin present ($(git remote get-url origin))"
else
  if [[ -z "$REMOTE_URL" ]]; then
    echo "ERROR: REMOTE_URL is required on first run (env var)."
    exit 2
  fi
  echo "REMOTE: add origin -> $REMOTE_URL"
  git remote add origin "$REMOTE_URL"
fi

# ---- stage & commit ----------------------------------------------------------
declare -a ADD
[[ -d configs ]] && ADD+=("configs")
# optional extras via env space-separated list
if [[ -n "${ADD_PATHS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${ADD_PATHS} )
  ADD+=("${EXTRA[@]}")
fi

if ((${#ADD[@]}==0)); then
  echo "WARN: nothing to add"
else
  git add "${ADD[@]}"
fi

if ! git diff --cached --quiet; then
  git commit -m "${COMMIT_MSG} (${STAMP})"
else
  echo "INFO: no changes staged; skipping commit"
fi

# ---- push --------------------------------------------------------------------
git push -u origin "$BRANCH"
echo "DONE: pushed ${ADD[*]:-nothing} to $BRANCH @ origin"
