SpectraMind V50 — configs/symbolic/overrides/regions

Hydra-compatible overlays controlling which symbolic region packs are active and how they are weighted.

Files
- default.yaml — Safe baseline; neutral weights; schema validation = warn.
- molecules.yaml — Emphasizes molecular fingerprints (H2O/CO2/CH4/NH3).
- telescope.yaml — Emphasizes channel/band integrity and cross-band continuity.
- astrophysics.yaml — Emphasizes continuum/slope/thermal behavior.
- challenge.yaml — Enables stress-test composites; strict validation; export to challenge bundle.
- local.yaml — Fast-iter dev mode; JSON masks only; warnings not fatal.
- kaggle.yaml — CI/leaderboard mode; strict; JSON masks to submission_bundle/.

Usage

Select an overlay via your Hydra defaults (example):

```yaml
# configs/config_v50.yaml (excerpt)
defaults:
  - symbolic/overrides/regions: default
```

Swap to other overlays by changing default → molecules, telescope, astrophysics, challenge,
local, or kaggle.

All packs reference canonical sources in configs/symbolic/regions/*.yaml and validate entries with
configs/symbolic/_schemas/symbolic_region.schema.json.
