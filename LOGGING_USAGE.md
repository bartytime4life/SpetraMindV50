# SpectraMind V50 – Logging via Hydra

Select a logging backend at runtime:
- Console:  `python your_entrypoint.py logging=console`
- File:     `python your_entrypoint.py logging=file`
- JSONL:    `python your_entrypoint.py logging=jsonl`
- MLflow:   `python your_entrypoint.py logging=mlflow`

In your Python entrypoint:

```python
from omegaconf import OmegaConf
from hydra import initialize, compose
from src.spectramind.logging.apply import install_from_hydra

# ... obtain `cfg` via Hydra as you normally do ...
install_from_hydra(cfg)
```

File logging writes `logs/v50_debug_log.md` (rotating). JSONL logging appends to `logs/v50_event_log.jsonl`.
