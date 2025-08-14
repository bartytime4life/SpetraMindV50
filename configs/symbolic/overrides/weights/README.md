# Symbolic Weights Overrides

This directory defines **validated** weight override layers used by SpectraMindV50’s symbolic stack.
Each `*.yaml` is validated against `schema.json` by `validate.py` (or `scripts/validate-weights.sh`).

## Files
- `schema.json` — JSON Schema for weight maps (namespaced keys, numeric values, optional bounds/notes).
- `default.yaml` — Conservative defaults suitable for CI and baseline runs.
- `strict.yaml` — Stricter weights (heavier regularization/penalties).
- `dev.yaml` — Developer-friendly weights with verbose logging & relaxed constraints.
- `version.txt` — Human-maintained semver for this override set.
- `validate.py` — Validation & instrumentation with:
  - rich console + rotating file logs
  - JSONL event stream (`logs/weights_events.jsonl`)
  - Git+ENV capture snapshot
  - Optional MLflow/W&B sync (toggle via env vars)

## Usage
- Validate all profiles:
  ```bash
  ./scripts/validate-weights.sh
  ```

- Validate a single profile:

```bash
python configs/symbolic/overrides/weights/validate.py --file configs/symbolic/overrides/weights/strict.yaml
```

- Enable trackers:

```bash
export WEIGHTS_TRACK_MLFLOW=1           # optional
export WEIGHTS_MLFLOW_EXPERIMENT="weights-overrides"
export WEIGHTS_MLFLOW_TRACKING_URI="file:./mlruns"

export WEIGHTS_TRACK_WANDB=1            # optional
export WANDB_PROJECT="weights-overrides"
```

Design notes
- Documentation-first & reproducible: configs + schema + logs + version markers are kept together for auditability.
- Minimal deps; MLflow/W&B are optional and only imported if env enables them.
- Logging is “Hydra-safe”: no duplicate handlers, and quiet by default unless --verbose.
