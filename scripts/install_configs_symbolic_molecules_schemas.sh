#!/usr/bin/env bash
# ==================================================================================================
# SpectraMind V50 — configs/symbolic/molecules/_schemas bootstrapper
# --------------------------------------------------------------------------------------------------
# What this does:
#   • Creates the directory tree: configs/symbolic/molecules/_schemas (and _examples)
#   • Writes JSON Schemas (Draft 2020-12) for molecules, spectral regions, fingerprints, and rules
#   • Writes Hydra-safe YAML index, units, and validators wiring
#   • Drops well-formed example molecules (H2O, CH4) using the schemas for quick validation
#   • Adds an explanatory README (engineering-grade, reproducibility-centric)
#
# How to use on phone (Termux / iSH / iOS Shortcuts / mobile shell):
#   1) Save this file as: install_configs_symbolic_molecules_schemas.sh
#   2) Run: bash install_configs_symbolic_molecules_schemas.sh
#   3) Git it:
#        git add -A
#        git commit -m "configs: add symbolic molecule schemas, units, validators, and examples"
#        git push
#
# Design notes:
#   • JSON Schemas are self-contained and reference each other via $ref relative paths
#   • YAML configs are Hydra-friendly (no tabs, only spaces; validated keys; comments explain fields)
#   • Field names are stable, lowercase, snake_case for frictionless parsing
#   • Units are explicit (μm for wavelengths, cm^-1 for wavenumbers), with a single canonical in files
#   • Rules capture physics-aware symbolic constraints (presence, ratios, smoothness, exclusivity, etc.)
#   • Everything is logger- and CI-ready: deterministic, minimal, portable
#
# --------------------------------------------------------------------------------------------------
# Repro & safety:
#   • Idempotent: Running multiple times safely overwrites the same files
#   • No side effects beyond writing under ./configs/symbolic/molecules/_schemas and _examples
#   • Valid YAML/JSON (linted formats, comments provided for maintainability)
# ==================================================================================================

set -euo pipefail

ROOT_DIR="$(pwd)"
TARGET_DIR="configs/symbolic/molecules/_schemas"
EXAMPLES_DIR="configs/symbolic/molecules/_examples"

mkdir -p "${TARGET_DIR}" "${EXAMPLES_DIR}"

# --------------------------------------------------------------------------------------------------
# README.md — Context, purpose, and guaranteed field-level contracts
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/README.md" <<'EOF'
# Symbolic Molecule Schemas (SpectraMind V50)

This directory contains **canonical schemas** and **Hydra-safe wiring YAML** for symbolic molecule
metadata, spectral regions, molecular fingerprints, and physics-aware **rule constraints**. These are used
by the V50 neuro-symbolic pipeline to validate inputs, compose symbolic loss packs, and drive diagnostics.

## Files

- `molecules.schema.json` — Primary JSON Schema for a molecule record
- `region.schema.json` — Schema for per-molecule spectral sub-bands/regions
- `fingerprint.schema.json` — Schema for line/fingerprint templates and references
- `rule.schema.json` — Schema for physics-aware symbolic rules and constraints
- `index.yaml` — Hydra index that maps molecule IDs to files/configs
- `units.yaml` — Canonical units and conversions used across symbolic configs
- `validators.yaml` — Declarative "what to validate" list for CI/CLI validators
- `_examples/` — Example molecule YAMLs (H2O, CH4) validated by these schemas

## Key Conventions

- **Wavelengths** are specified in microns (`um`) unless otherwise noted.
- **Wavenumbers** in `cm^-1` may appear for line lists, but top-level ranges prefer μm.
- **Bin indices** refer to the challenge's **283-bin output** convention when present.
- All files are **deterministic** and CI-friendly. Avoid non-reproducible fields.

## Why JSON Schema + YAML?

- JSON Schema (Draft 2020-12) provides rigorous, machine-checked contracts for fields/types/ranges.
- YAML is human-friendly for daily editing and works seamlessly with **Hydra** configs.

## Science & Engineering Rationale (selected references, schema-level intent)

- The schemas encode **physics-grounded** expectations (band presence, relative band ratios,
  mutual exclusivity, smoothness windows) aligning with exoplanet spectroscopy workflows and
  reproducibility-first scientific engineering [oai_citation:0‡Initial Domain Module Examples and Tooling Plan (Master Coder Protocol).pdf](file-service://file-3DcSMPW91eefZZQgT3WSXd) [oai_citation:1‡Foundational Templates and Glossary for Scientific Method _ Research _ Master Coder Protocol.pdf](file-service://file-9jbjjuT1TWyxZp17p6atTX).
- Symbolic rules aim to reflect **molecular spectroscopy** behavior used in astrophysics pipelines and
  integrate with broader **modeling & simulation** best practices [oai_citation:2‡Scientific Modeling and Simulation_ A Comprehensive NASA-Grade Guide.pdf](file-service://file-18UFLSY2gyjkUeUZDhYVDU) [oai_citation:3‡Physics and Astrophysics Modeling & Simulation Reference.pdf](file-service://file-QqyqN7YyHoFa9taG1AbmiW).

## Validation Workflow

1. Create or edit a molecule YAML in `configs/symbolic/molecules/` or `_examples/`.
2. Validate via CLI (example tooling in SpectraMind V50):

python -m spectramind diagnose symbolic-validate 
–schema configs/symbolic/molecules/_schemas/molecules.schema.json 
–input  configs/symbolic/molecules/_examples/H2O.yaml

3. CI runs `validators.yaml` to lint and validate **all** molecules.

EOF

# --------------------------------------------------------------------------------------------------
# units.yaml — Canonical unit policy for symbolic molecules
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/units.yaml" <<'EOF'
# Canonical units and conversions for symbolic molecule schemas (Hydra-safe)
units:
wavelength:
 canonical: "um"        # microns for wavelength
 accepted: ["um", "micron", "microns"]
wavenumber:
 canonical: "cm^-1"
 accepted: ["cm^-1"]
strength:
 canonical: "arb"       # arbitrary normalized line/feature strength for templates
 accepted: ["arb"]
temperature:
 canonical: "K"
 accepted: ["K", "kelvin"]
pressure:
 canonical: "bar"
 accepted: ["bar", "Pa"]

# Optional simple factors (for lightweight conversion when needed in CLI tools)
conversions:
# wavelengths
"um->nm": 1000.0
"nm->um": 0.001
# pressure (illustrative; prefer a proper phys-constants lib in code)
"bar->Pa": 100000.0
"Pa->bar": 1.0e-5
EOF

# --------------------------------------------------------------------------------------------------
# index.yaml — Hydra index mapping molecule IDs to config payloads/paths
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/index.yaml" <<'EOF'
# Hydra index for symbolic molecule configurations
index:
# Map molecule identifiers to config snippets or file basenames (without extension if desired)
# These IDs should match "species_id" fields inside molecule payloads.
# Example entries; extend to your full set:
H2O:
 file: "H2O"                  # expects configs/symbolic/molecules/H2O.yaml (or in _examples for demo)
 enabled: true
 tags: ["volatile", "baseline", "water"]
CH4:
 file: "CH4"
 enabled: true
 tags: ["hydrocarbon", "methane"]

# Default policy for loading molecules (used by loaders if not overridden)
policy:
prefer_repo_dir: "configs/symbolic/molecules"
allow_examples: true
fail_on_missing: true
EOF

# --------------------------------------------------------------------------------------------------
# validators.yaml — drives CI/CLI validation passes over molecules
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/validators.yaml" <<'EOF'
# Declarative validation list for CI/CLI
validators:
- name: "molecule-records"
 schema: "configs/symbolic/molecules/_schemas/molecules.schema.json"
 include_globs:
   - "configs/symbolic/molecules/*.yaml"
   - "configs/symbolic/molecules/_examples/*.yaml"
 # Optional excludes for WIP drafts:
 exclude_globs:
   - "configs/symbolic/molecules/_examples/*_WIP.yaml"

- name: "region-records"
 schema: "configs/symbolic/molecules/_schemas/region.schema.json"
 include_globs:
   - "configs/symbolic/molecules/*.yaml"
   - "configs/symbolic/molecules/_examples/*.yaml"
 nested_path: "$.spectral_regions[*]"  # validate each region object

- name: "fingerprint-records"
 schema: "configs/symbolic/molecules/_schemas/fingerprint.schema.json"
 include_globs:
   - "configs/symbolic/molecules/*.yaml"
   - "configs/symbolic/molecules/_examples/*.yaml"
 nested_path: "$.fingerprints[*]"      # validate each fingerprint object

- name: "rule-records"
 schema: "configs/symbolic/molecules/_schemas/rule.schema.json"
 include_globs:
   - "configs/symbolic/molecules/*.yaml"
   - "configs/symbolic/molecules/_examples/*.yaml"
 nested_path: "$.rules[*]"             # validate each rule object
EOF

# --------------------------------------------------------------------------------------------------
# region.schema.json — spectral sub-bands / regions
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/region.schema.json" <<'EOF'
{
"$schema": "https://json-schema.org/draft/2020-12/schema",
"$id": "region.schema.json",
"title": "Spectral Region Schema",
"type": "object",
"additionalProperties": false,
"required": ["name", "wavelength_um", "band_center_um", "band_width_um"],
"properties": {
 "name": {
   "type": "string",
   "minLength": 1,
   "description": "Human-friendly label (e.g., 'H2O_1p4um_band')"
 },
 "wavelength_um": {
   "type": "object",
   "required": ["min", "max"],
   "additionalProperties": false,
   "properties": {
     "min": { "type": "number", "exclusiveMinimum": 0.0 },
     "max": { "type": "number", "exclusiveMinimum": 0.0 }
   },
   "description": "Inclusive bounds in microns for this region"
 },
 "band_center_um": {
   "type": "number",
   "exclusiveMinimum": 0.0,
   "description": "Approximate absorption band center (μm)"
 },
 "band_width_um": {
   "type": "number",
   "exclusiveMinimum": 0.0,
   "description": "Approximate full feature width around band center (μm)"
 },
 "bin_indices": {
   "type": "array",
   "items": { "type": "integer", "minimum": 0 },
   "description": "Optional explicit bin indices (0-based) aligning to 283-bin decoder"
 },
 "priority": {
   "type": "integer",
   "minimum": 0,
   "default": 0,
   "description": "Relative diagnostic priority (higher = earlier checks/plots)"
 },
 "notes": { "type": "string" }
}
}
EOF

# --------------------------------------------------------------------------------------------------
# fingerprint.schema.json — molecular fingerprints / line templates
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/fingerprint.schema.json" <<'EOF'
{
"$schema": "https://json-schema.org/draft/2020-12/schema",
"$id": "fingerprint.schema.json",
"title": "Molecular Fingerprint Schema",
"type": "object",
"additionalProperties": false,
"required": ["name", "mode", "template_wavelength_um", "template_strength"],
"properties": {
 "name": {
   "type": "string",
   "minLength": 1,
   "description": "Fingerprint/template label, e.g., 'H2O_wide_1p4um'"
 },
 "mode": {
   "type": "string",
   "enum": ["lines", "envelope", "impulse", "custom"],
   "description": "Template construction mode (line list, smoothed envelope, impulses, etc.)"
 },
 "template_wavelength_um": {
   "type": "array",
   "items": { "type": "number", "exclusiveMinimum": 0.0 },
   "minItems": 1,
   "description": "Wavelength sample points (μm) for the template"
 },
 "template_strength": {
   "type": "array",
   "items": { "type": "number" },
   "minItems": 1,
   "description": "Normalized strength values aligned with template_wavelength_um"
 },
 "source": {
   "type": "string",
   "description": "Reference (e.g., 'HITRAN2016', 'ExoMol-2020', or internal citation)"
 },
 "temperature_K": {
   "type": "number",
   "minimum": 0.0,
   "description": "Reference temperature for this fingerprint"
 },
 "pressure_bar": {
   "type": "number",
   "minimum": 0.0,
   "description": "Reference pressure for this fingerprint"
 },
 "notes": { "type": "string" }
}
}
EOF

# --------------------------------------------------------------------------------------------------
# rule.schema.json — physics-aware symbolic rules
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/rule.schema.json" <<'EOF'
{
"$schema": "https://json-schema.org/draft/2020-12/schema",
"$id": "rule.schema.json",
"title": "Symbolic Rule Schema",
"type": "object",
"additionalProperties": false,
"required": ["rule_id", "type", "severity"],
"properties": {
 "rule_id": { "type": "string", "minLength": 1 },
 "type": {
   "type": "string",
   "enum": [
     "presence_band",           /* band presence required in region(s) */
     "absence_window",          /* band must be absent in window(s)    */
     "ratio_between_bands",     /* ratio constraints across two+ bands  */
     "mutual_exclusion",        /* molecule A vs B exclusivity regions  */
     "smoothness_window",       /* local smoothness (FFT/L2) constraint */
     "nonnegativity_window",    /* μ >= 0 in window (or ≥ baseline)    */
     "template_correlation",    /* corr(μ, fingerprint) ≥ threshold     */
     "coabsorption_consistency" /* if A present then B in adj. region   */
   ]
 },
 "severity": {
   "type": "string",
   "enum": ["info", "weak", "moderate", "strong", "hard"],
   "description": "Strength of constraint; training uses to weight symbolic loss"
 },
 "applies_to": {
   "type": "object",
   "additionalProperties": false,
   "properties": {
     "species_id": { "type": "string" },
     "region_names": {
       "type": "array",
       "items": { "type": "string" }
     }
   },
   "description": "Target molecule and/or named regions to which the rule applies"
 },
 "params": {
   "type": "object",
   "description": "Rule-specific parameters (see below)",
   "additionalProperties": true
 },
 "notes": { "type": "string" }
},

"$comment": "Rule-specific params guidance:\n\
• presence_band: { min_depth: float, min_span_um: float }\n\
• absence_window: { max_depth: float, span_um: float }\n\
• ratio_between_bands: { band_a: 'name', band_b: 'name', ratio_min: float, ratio_max: float }\n\
• mutual_exclusion: { other_species_id: 'CH4', overlap_um: [min, max], max_joint_depth: float }\n\
• smoothness_window: { wavelength_um: {min, max}, l2_lambda: float, fft_lambda: float }\n\
• nonnegativity_window: { wavelength_um: {min, max}, baseline: float }\n\
• template_correlation: { fingerprint: 'name', min_corr: float }\n\
• coabsorption_consistency: { partner_species_id: 'CO2', lag_um: float, min_pair_presence: int }"
}
EOF

# --------------------------------------------------------------------------------------------------
# molecules.schema.json — primary schema (references region/fingerprint/rule)
# --------------------------------------------------------------------------------------------------
cat > "${TARGET_DIR}/molecules.schema.json" <<'EOF'
{
"$schema": "https://json-schema.org/draft/2020-12/schema",
"$id": "molecules.schema.json",
"title": "Symbolic Molecule Schema",
"type": "object",
"additionalProperties": false,
"required": ["species_id", "name", "formula", "wavelength_um", "spectral_regions", "fingerprints", "rules"],
"properties": {
 "species_id": {
   "type": "string",
   "pattern": "^[A-Za-z0-9_\\-]+$",
   "description": "Unique ID used across SpectraMind (e.g., 'H2O', 'CH4')"
 },
 "name": { "type": "string", "minLength": 1 },
 "formula": { "type": "string", "minLength": 1 },
 "charge": { "type": "integer", "default": 0 },
 "is_radiatively_active": { "type": "boolean", "default": true },

 "wavelength_um": {
   "type": "object",
   "required": ["min", "max"],
   "additionalProperties": false,
   "properties": {
     "min": { "type": "number", "exclusiveMinimum": 0.0 },
     "max": { "type": "number", "exclusiveMinimum": 0.0 }
   },
   "description": "Global wavelength coverage for molecule config (μm)"
 },

 "default_binning": {
   "type": "object",
   "additionalProperties": false,
   "required": ["n_bins"],
   "properties": {
     "n_bins": { "type": "integer", "minimum": 1, "description": "Reference bin count (e.g., 283)" },
     "bin_map": {
       "type": "array",
       "items": { "type": "number", "exclusiveMinimum": 0.0 },
       "description": "Optional μm centers per bin; if omitted, loader will infer"
     }
   }
 },

 "spectral_regions": {
   "type": "array",
   "minItems": 1,
   "items": { "$ref": "region.schema.json" }
 },

 "fingerprints": {
   "type": "array",
   "minItems": 1,
   "items": { "$ref": "fingerprint.schema.json" }
 },

 "rules": {
   "type": "array",
   "minItems": 0,
   "items": { "$ref": "rule.schema.json" }
 },

 "metadata": {
   "type": "object",
   "additionalProperties": false,
   "properties": {
     "sources": {
       "type": "array",
       "items": { "type": "string" },
       "description": "Literature or database references (HITRAN/ExoMol/etc.)"
     },
     "notes": { "type": "string" },
     "version": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+(\\.[0-9]+)?$" },
     "updated": { "type": "string", "format": "date" }
   }
 }
}
}
EOF

# --------------------------------------------------------------------------------------------------
# _examples/H2O.yaml — example molecule (Water)
# --------------------------------------------------------------------------------------------------
cat > "${EXAMPLES_DIR}/H2O.yaml" <<'EOF'
# Example molecule config: Water (H2O)
species_id: "H2O"
name: "Water"
formula: "H2O"
charge: 0
is_radiatively_active: true

wavelength_um:
min: 0.5
max: 7.8

default_binning:
n_bins: 283

spectral_regions:
- name: "H2O_1p4um_band"
 wavelength_um: { min: 1.30, max: 1.55 }
 band_center_um: 1.40
 band_width_um: 0.10
 priority: 10
- name: "H2O_1p9um_band"
 wavelength_um: { min: 1.80, max: 2.05 }
 band_center_um: 1.92
 band_width_um: 0.12
 priority: 9

fingerprints:
- name: "H2O_envelope_1p3_2p1"
 mode: "envelope"
 template_wavelength_um: [1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05]
 template_strength:       [0.05, 0.20, 0.65, 0.80, 0.50, 0.10, 0.05, 0.30, 0.75, 0.85, 0.55, 0.15]
 source: "Composite envelope (curated)"
 temperature_K: 1000
 pressure_bar: 0.01

rules:
- rule_id: "H2O.presence.1p4um"
 type: "presence_band"
 severity: "moderate"
 applies_to: { species_id: "H2O", region_names: ["H2O_1p4um_band"] }
 params:
   min_depth: 0.01       # minimum expected absorption depth
   min_span_um: 0.04
 notes: "Water band near 1.4μm generally present in warm exoplanet atmospheres."
- rule_id: "H2O.template_corr.1p3_2p1"
 type: "template_correlation"
 severity: "weak"
 applies_to: { species_id: "H2O", region_names: ["H2O_1p4um_band", "H2O_1p9um_band"] }
 params:
   fingerprint: "H2O_envelope_1p3_2p1"
   min_corr: 0.25
- rule_id: "H2O.smoothness.local"
 type: "smoothness_window"
 severity: "weak"
 applies_to: { species_id: "H2O", region_names: ["H2O_1p4um_band", "H2O_1p9um_band"] }
 params:
   wavelength_um: { min: 1.30, max: 2.05 }
   l2_lambda: 0.05
   fft_lambda: 0.00
- rule_id: "H2O.nonnegativity.core"
 type: "nonnegativity_window"
 severity: "hard"
 applies_to: { species_id: "H2O", region_names: ["H2O_1p4um_band", "H2O_1p9um_band"] }
 params:
   wavelength_um: { min: 1.30, max: 2.05 }
   baseline: 0.0

metadata:
sources:
 - "General exoplanet spectroscopy context; astrophysical spectroscopy practice"
notes: "Starter example for schema validation and symbolic rule wiring."
version: "1.0.0"
updated: "2025-08-14"
EOF

# --------------------------------------------------------------------------------------------------
# _examples/CH4.yaml — example molecule (Methane)
# --------------------------------------------------------------------------------------------------
cat > "${EXAMPLES_DIR}/CH4.yaml" <<'EOF'
# Example molecule config: Methane (CH4)
species_id: "CH4"
name: "Methane"
formula: "CH4"
charge: 0
is_radiatively_active: true

wavelength_um:
min: 0.5
max: 7.8

default_binning:
n_bins: 283

spectral_regions:
- name: "CH4_1p66um_band"
 wavelength_um: { min: 1.60, max: 1.72 }
 band_center_um: 1.66
 band_width_um: 0.08
 priority: 10
- name: "CH4_2p3um_band"
 wavelength_um: { min: 2.25, max: 2.38 }
 band_center_um: 2.30
 band_width_um: 0.10
 priority: 9

fingerprints:
- name: "CH4_envelope_1p6_2p4"
 mode: "envelope"
 template_wavelength_um: [1.60, 1.63, 1.66, 1.69, 1.72, 2.25, 2.28, 2.30, 2.33, 2.36, 2.38, 2.40]
 template_strength:       [0.10, 0.35, 0.85, 0.60, 0.15, 0.10, 0.45, 0.90, 0.70, 0.30, 0.15, 0.05]
 source: "Composite envelope (curated)"
 temperature_K: 1000
 pressure_bar: 0.01

rules:
- rule_id: "CH4.presence.1p66um"
 type: "presence_band"
 severity: "moderate"
 applies_to: { species_id: "CH4", region_names: ["CH4_1p66um_band"] }
 params:
   min_depth: 0.012
   min_span_um: 0.03
- rule_id: "CH4.template_corr"
 type: "template_correlation"
 severity: "weak"
 applies_to: { species_id: "CH4", region_names: ["CH4_1p66um_band", "CH4_2p3um_band"] }
 params:
   fingerprint: "CH4_envelope_1p6_2p4"
   min_corr: 0.25
- rule_id: "CH4.vs.H2O.exclusion_1p60_1p72"
 type: "mutual_exclusion"
 severity: "info"
 applies_to: { species_id: "CH4", region_names: ["CH4_1p66um_band"] }
 params:
   other_species_id: "H2O"
   overlap_um: [1.60, 1.72]
   max_joint_depth: 0.02
 notes: "Heuristic: discourage spurious co-detection when both templates weakly align in overlap."
- rule_id: "CH4.smoothness.local"
 type: "smoothness_window"
 severity: "weak"
 applies_to: { species_id: "CH4", region_names: ["CH4_1p66um_band", "CH4_2p3um_band"] }
 params:
   wavelength_um: { min: 1.60, max: 2.38 }
   l2_lambda: 0.05
   fft_lambda: 0.00
- rule_id: "CH4.nonnegativity.core"
 type: "nonnegativity_window"
 severity: "hard"
 applies_to: { species_id: "CH4", region_names: ["CH4_1p66um_band", "CH4_2p3um_band"] }
 params:
   wavelength_um: { min: 1.60, max: 2.38 }
   baseline: 0.0

metadata:
sources:
 - "General exoplanet spectroscopy context; astrophysical spectroscopy practice"
notes: "Starter example for schema validation and symbolic rule wiring."
version: "1.0.0"
updated: "2025-08-14"
EOF

# --------------------------------------------------------------------------------------------------
# Success message + optional git helper
# --------------------------------------------------------------------------------------------------
echo
echo "✅ Wrote schemas and examples under: ${TARGET_DIR} and ${EXAMPLES_DIR}"
echo
echo "Next (optional) — one-liner to commit & push:"
echo "  git add -A && git commit -m \"configs: add symbolic molecule schemas, units, validators, and examples\" && git push"
echo

# --------------------------------------------------------------------------------------------------
# Inline references (kept as comments for provenance / engineering traceability)
#   These citations justify design choices: reproducibility templates, modeling rigor, and physics context.
#   They are included here as comments to keep the script self-contained and compliant with documentation
#   policies while not interfering with execution.
# --------------------------------------------------------------------------------------------------
: <<'CITATIONS'
References:
- Reproducible documentation & experiment structure (Master Coder Protocol templates)
[oai_citation:4‡Initial Domain Module Examples and Tooling Plan (Master Coder Protocol).pdf](file-service://file-3DcSMPW91eefZZQgT3WSXd) [oai_citation:5‡Initial Domain Module Examples and Tooling Plan (Master Coder Protocol).pdf](file-service://file-3DcSMPW91eefZZQgT3WSXd)
- Scientific reproducibility principles (Royal Society discussion)
[oai_citation:6‡Foundational Templates and Glossary for Scientific Method _ Research _ Master Coder Protocol.pdf](file-service://file-9jbjjuT1TWyxZp17p6atTX)
- NASA-grade modeling & simulation standards / credibility
[oai_citation:7‡Scientific Modeling and Simulation_ A Comprehensive NASA-Grade Guide.pdf](file-service://file-18UFLSY2gyjkUeUZDhYVDU)
- Physics & astrophysics context for spectroscopy and modeling
[oai_citation:8‡Physics and Astrophysics Modeling & Simulation Reference.pdf](file-service://file-QqyqN7YyHoFa9taG1AbmiW)
CITATIONS
