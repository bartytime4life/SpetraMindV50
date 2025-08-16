# SpectraMind V50 — Logging

Mission-grade logging for the NeurIPS 2025 Ariel Data Challenge.

* Console + rotating file logs
* JSONL event stream (`logs/events.jsonl`)
* Hydra-safe config via `LoggingConfig` and templates in `hydra_templates/`
* Optional MLflow / W&B
* Telemetry helpers
* CI/self-test validators

Quick start:

```python
from spectramind.logging import LoggingConfig, init_logging, get_logger
cfg = LoggingConfig(log_level="DEBUG", log_dir="logs", jsonl=True)
init_logging(cfg)
log = get_logger("spectramind.demo")
log.info("Mission logging online", extra={"component": "demo"})
```

Hydra templates are provided in `hydra_templates/`. Copy or adapt them to your
configs tree (e.g. `configs/hydra/job_logging`).
