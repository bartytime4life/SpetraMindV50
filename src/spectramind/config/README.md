# SpectraMind V50 Config System

Mission-grade configuration layer for the SpectraMind V50 pipeline.

## What you get

- **Schema-enforced** YAML/JSON configs (`config_schema.yaml`, `config_validator.py`)
- **Hydra/OmegaConf compatible** overrides (`load_hydra_config`, `hydra_overrides.yaml`)
- **Auto-registration on import**:
  - `defaults` → `defaults.yaml`
  - `v50` → `config_v50.yaml`
- **Global registry** (`registry.py`) available as `spectramind.config.CONFIG_REGISTRY`
- **Environment helpers** (`env.py`) with CUDA-aware device detection

## Quick start

Importing the package auto-registers configs:

```python
from spectramind.config import CONFIG_REGISTRY, get_config  # get_config via registry if you import it
# Access auto-registered configs
v50 = CONFIG_REGISTRY["v50"]
defaults = CONFIG_REGISTRY["defaults"]
```

Programmatic load + validation from a custom path:

```python
from spectramind.config import load_config
cfg = load_config("path/to/your_config.yaml")
```

Hydra/OmegaConf-style build from overrides:

```python
from spectramind.config import load_hydra_config
cfg = load_hydra_config({"training": {"epochs": 100, "batch_size": 64}})
```

## CLI/Hydra tip

You can pass overrides when launching your scripts (if they parse OmegaConf/Hydra inputs):

```bash
python train_v50.py model.encoder=fgs1_mamba training.epochs=100
```

## Validation

All loads validate against `config_schema.yaml` using `jsonschema`. Install:

```bash
pip install jsonschema pyyaml omegaconf
# (hydra-core optional for full Hydra workflows)
```

## Silence import logs

Set `SPECTRAMIND_CONFIG_SILENT=1` to disable the one-line auto-registration message:

```bash
export SPECTRAMIND_CONFIG_SILENT=1
```

