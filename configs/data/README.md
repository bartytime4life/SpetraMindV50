# /configs/data

Hydra YAML configs for dataset paths, formats, and loader parameters.

## Files
- **base.yaml** — shared defaults for all environments.
- **local.yaml** — paths for local workstation runs.
- **kaggle.yaml** — paths for Kaggle competition container.
- **schema.yaml** — optional schema/shape definitions for integrity checks.
- **README.md** — this file.

These configs are loaded by `spectramind.py` commands via Hydra.
Override on CLI, e.g.:

```bash
python -m spectramind calibrate data=local
python -m spectramind train data=kaggle