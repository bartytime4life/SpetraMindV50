# Symbolic Profiles (Overrides)

Mission: Provide declarative, Hydra-safe symbolic profile overlays that select physically grounded rule packs and weights per planet/instrument context, and can be switched dynamically (by metadata, events, or CLI).

## Why profiles?
- Different planet classes (UHJ, HJ, Warm Neptune, Temperate Super-Earth) exhibit distinct spectral fingerprints and systematics; profiles tune symbolic constraints (smoothness, nonnegativity, molecule-region priors, FFT windows, photonic alignment, quantile/σ calibration weights) accordingly.
- Profiles also gate calibration & decoder knobs for instrument-specific behavior (FGS1 vs. AIRS), while remaining compliant with reproducibility and CI schema validation.

## Loading order
1. `index.yaml` declares the active profile and lists available profiles.
2. `profile_map.yaml` contains rule-based selectors that can auto-switch the profile based on metadata (e.g., T_eq, log g).
3. `*.profile.yaml` files define self-contained overlays merged by Hydra into the runtime config tree.

## Validation
- JSON Schema-ish documents live in `../_schemas/`. CI/selftest validates structure (required keys/ranges).
- Keep comments; they’re stripped by loaders before validation if needed.

## Integration
- `../instruments/link_profiles.yaml` lets instrument overrides reference the active profile class for consistent weights.
- `../events/profile_switch.rules.yaml` offers declarative event-driven switches (e.g., "if OOD_snr_drop > τ → use ood_guardrails").
- `../competition/profiles.yaml` can pin profiles in "leaderboard mode".
