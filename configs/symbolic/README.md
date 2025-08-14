# SpectraMind V50 — configs/symbolic Guide

## How to use
- Default entrypoint: `symbolic/base.yaml`
- Compose with Hydra:
  - `+symbolic.profile=leaderboard`
  - `+symbolic.weights=strict`
  - `+symbolic.regions=airs_regions`
- Add/remove rules by editing defaults in `base.yaml` or via command-line group overrides.

## Key files
- `rules/*.yaml` — individual rule configs merged under `symbolic.rules.*`
- `profiles/*.yaml` — scenario presets; may override weights/toggles/caps
- `weights/*.yaml` — per-rule weights collections
- `molecules/*.yaml` — band templates (bins and/or wavelengths)
- `regions/*.yaml` — named bin ranges used by rules for masking/boosting
- `export.yaml` — output artifact locations
- `debug.yaml` — logging verbosity and sampling
- `schema/symbolic.schema.json` — sanity schema for CI checks

## Replace placeholders
- Update `regions/airs_regions.yaml` and `molecules/lines_h2o_co2_ch4.yaml` with dataset-accurate bin or wavelength ranges once your wavelength↔bin CSV is finalized.
