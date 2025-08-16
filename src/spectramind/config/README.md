# SpectraMind V50 Config System

This directory contains **Hydra-compatible configuration files** for the SpectraMind V50 pipeline.

## Structure
- `defaults.yaml` – Global defaults, reproducibility, experiment metadata
- `model.yaml` – Encoder/decoder/symbolic model settings
- `training.yaml` – Optimizer, scheduler, trainer, loss
- `calibration.yaml` – Temperature scaling, COREL calibration
- `diagnostics.yaml` – FFT, UMAP, t-SNE, SHAP, symbolic overlays
- `logging.yaml` – Console, file, JSONL, MLflow logging
- `schema.py` – Pydantic schema for config validation
- `validator.py` – Schema-based validator
- `registry.py` – Unified loader/validator for Hydra configs

## Usage
Run with Hydra:
```bash
python train_v50.py --config-path src/spectramind/config --config-name defaults
```

Validate a config:

```bash
python -m src.spectramind.config.validator src/spectramind/config/model.yaml
```

Load programmatically:

```python
from src.spectramind.config.registry import load_config
cfg = load_config("training")
```

## Mission Readiness

* ✅ Hydra group defaults
* ✅ Pydantic schema validation
* ✅ Logging + MLflow integration
* ✅ Symbolic-aware calibration and diagnostics
* ✅ CI/CLI compatibility
