# configs/symbolic/overrides

This directory contains override packs that specialize the V50 symbolic system per planet, instrument, run-mode, or data quality. These YAMLs are composed via Hydra and routed using `routing.yaml` match rules on metadata (e.g., `meta.teq`, `meta.jitter_rms_px`, `meta.snr_db`).

## Quick use
- Default add-on: `+configs/symbolic/overrides/base`
- Apply profiles: `+configs/symbolic/overrides/profiles/profile_high_jitter`
- Add instrument overrides: `+configs/symbolic/overrides/instruments/fgs1_strict`
- Auto-route by metadata: include `+configs/symbolic/overrides/routing` (engine selects).

## Structure
- `_schemas/overrides.schema.yaml` — soft schema to sanity-check keys.
- `base.yaml` — conservative defaults (safe everywhere).
- `routing.yaml` — metadata → override selection map (ordered rules).
- `profiles/` — planet/data quality profiles (`hot_jupiter`, `low_snr`, `high_haze`, etc.).
- `molecules/` — molecule-specific constraints & windows (`H2O`, `CO2`, `CH4`, `NH3`, `CO`).
- `instruments/` — per-instrument constraint adjustments (`FGS1` strict, `AIRS` relaxed).
- `events/` — transit vs eclipse specialization and cadence-aware toggles.
- `weights/` — hard/soft weighting presets for symbolic terms.
- `violations/` — allow/deny rule sets; emergency demotion/promotions during triage.
- `competition/` — Kaggle runtime guardrails & 9h budget protections.

## Log & diagnostics integration

These configs include:
- `events_jsonl.enable` to emit run-time JSONL events.
- `logs.rotate_mb` & `logs.max_files` to keep console + rotating file logs tidy.
- MLflow/W&B toggles via environment variables (no-ops if disabled).
- `diagnostics.flags.*` to surface overlays in the HTML dashboard.
