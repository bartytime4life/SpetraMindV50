# Competition Overrides (Hydra)

This folder contains mission-grade Hydra override packs tuned for the NeurIPS 2025 Ariel Data Challenge. They do not change base schema/structure—they only override values for symbolic constraints, calibration, uncertainty, runtime guardrails, and diagnostics.

## How to select an override

Pick one of these when you call the CLI (examples shown for Typer; works with `python -m spectramind` as well):

- **Leaderboard** (full fidelity, strict constraints & COREL):

```bash
python -m spectramind submit +symbolic/overrides=competition/leaderboard
```

- **Kaggle GPU** (9-hour guardrail, tight logging, compact diagnostics):

```bash
python -m spectramind submit +symbolic/overrides=competition/kaggle_gpu
```

- **Local GPU** (developer machine, more diagnostics):

```bash
python -m spectramind train +symbolic/overrides=competition/local_gpu
```

- **Fast Debug** (minutes, reduced weights, tiny batches, quick dashboard):

```bash
python -m spectramind diagnose dashboard +symbolic/overrides=competition/fast_debug
```

## Files

- `leaderboard.yaml` — max quality, stricter symbolic + COREL coverage targets, calibrated σ export.
- `kaggle_gpu.yaml` — 9h walltime guardrail, reduced dashboard payload, artifact quotas enforced.
- `local_gpu.yaml` — dev-friendly logging/diagnostics; higher verbosity.
- `fast_debug.yaml` — quick iteration; tiny data slices; soft symbolic weights.
- `rules_strict.yaml` — strict symbolic rule set (referenced by leaderboard).
- `rules_minimal.yaml` — minimal but sane rules; used by fast_debug.
- `profiles.yaml` — symbolic profile catalog (molecule bands, detector regions, priors).
- `uncertainty.yaml` — σ decoder and export policy overrides.
- `calibration.yaml` — temperature scaling + COREL options, coverage & reporting.
- `corel.yaml` — COREL GNN backend, edge features, positional encodings.
- `runtime.yaml` — guardrails, logging (JSONL + rotating files), artifact quotas, CI flags.

These work as value overlays. Your base configs remain the source of truth; choose exactly one of the competition packs to layer on top per run.
