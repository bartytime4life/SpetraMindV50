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

