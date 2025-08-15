# SpectraMind V50 — Symbolic Profiles

This package defines the Symbolic Profiles subsystem for the neuro-symbolic pipeline:

- YAML-defined profiles of rule weights/priorities/conditions
- Override/extends mechanism via `configs/symbolic/overrides/profiles`
- Mission-grade logging with JSONL event stream
- CLI for listing, validating, showing, diagnosing, activating, and exporting profiles

## Quickstart

```bash
python -m src.spectramind.symbolic.profiles.cli_profiles list
python -m src.spectramind.symbolic.profiles.cli_profiles show hot_jupiter_core
python -m src.spectramind.symbolic.profiles.cli_profiles activate warm_neptune_core
python -m src.spectramind.symbolic.profiles.cli_profiles diagnose --viol diagnostic_summary.json --out-dir reports/profiles
```

## Overrides

Place files under `configs/symbolic/overrides/profiles/*.yaml`. Example with `extends`:

```yaml
profiles:
  - id: hot_jupiter_strict
    extends: hot_jupiter_core
    name: Hot Jupiter — Strict
    description: Increases FFT discipline and molecule coherence.
    rules:
      - id: fft_continuum_suppress
        weight: 1.0
        priority: 5
```
