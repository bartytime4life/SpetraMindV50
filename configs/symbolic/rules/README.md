# SpectraMind V50 — Symbolic Rules

Each YAML file defines a single rule with:
- `rule.id` (unique), domain, kind, version, description
- `inputs` (tensor or resource names expected by the engine)
- `params` (tunable hyperparameters)
- `loss` (type/aggregation) and diagnostics (export/overlay flags)

Rule IDs referenced by profiles (e.g., `smoothness_constraint`) must match
the `rule.id` field here exactly.

## Folders
- `physics/`  — physical and signal-processing constraints
- `astro/`    — astronomy and transit‑spectroscopy constraints
- `biology/`  — cross-domain demonstration constraints
- `patterns/` — pattern-theoretic constraints

These integrate with:
- `symbolic_logic_engine.py`
- diagnostics HTML (`generate_html_report.py`)
- COREL/uncertainty calibration overlays
