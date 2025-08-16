# SpectraMind V50 Logging

Mission-grade logging system for the NeurIPS 2025 Ariel Data Challenge.

## Features
- Console + rotating file logs
- JSONL event stream (`logs/events.jsonl`)
- Hydra-safe configuration
- Optional MLflow / W&B sync
- Structured telemetry hooks

## Usage
```python
from spectramind.logging import init_logging, get_logger, LoggingConfig

cfg = LoggingConfig(log_level="DEBUG", mlflow=True, wandb=True)
init_logging(cfg)
log = get_logger("spectramind")

log.info("Mission logging online")
```

## Tests

Run `pytest` in this directory to validate logging system.
