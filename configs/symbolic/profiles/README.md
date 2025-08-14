# SpectraMind V50 — Symbolic Profiles

This directory contains Hydra YAML configs defining symbolic profiles for the
neuro‑symbolic reasoning engine. A profile is a named collection of symbolic
rules, weights, and domain tags used in training, calibration, and diagnostics.

## Files
- `base.yaml` — Canonical profile schema used by all profiles.
- `default_physics.yaml` — Physics-informed constraints.
- `default_astronomy.yaml` — Astronomy/transit‑spectroscopy constraints.
- `default_patterns.yaml` — Pattern & fractal‑theory constraints.
- `default_biology.yaml` — Cross-domain biological plausibility constraints.

## Usage

Select a profile via Hydra (examples):
- Python module:
  ```
  python -m spectramind train_v50 symbolic.profile=default_astronomy
  ```
- Typer CLI:
  ```
  spectramind train --config-name=config_v50 symbolic.profile=default_physics
  ```

## Notes
- Rule IDs listed in `rules.include` must exist under `configs/symbolic/rules/**`.
- Weights can be overridden per-rule via `weights.overrides`.
- Diagnostics flags integrate with SHAP overlays, HTML dashboards, and COREL.
