# Hydra Logging Templates – SpectraMind V50

This directory provides canonical logging dictConfig templates used across the pipeline.
Operational Hydra configs that import these patterns live in `configs/logging/`.

Backends:
- Console (human-readable CLI)
- File (rotating markdown log at `logs/v50_debug_log.md`)
- JSONL (structured event stream at `logs/v50_event_log.jsonl`)
- MLflow (console + MLflow tracking fields in cfg)

These are mirrored in `configs/logging/*.yaml` for direct Hydra selection with:
`logging=console`, `logging=file`, `logging=jsonl`, `logging=mlflow`.
