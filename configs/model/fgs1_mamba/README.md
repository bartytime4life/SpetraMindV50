# /configs/model/fgs1_mamba

Hydra YAML configs for the **FGS1 Mamba SSM encoder** used in SpectraMind V50.

These configs let you override just the `model.encoders.fgs1_mamba` section of the main `model/base.yaml`.

## Usage

Run with Hydra override syntax:
```bash
python -m spectramind train model.fgs1_mamba=base
python -m spectramind train model.fgs1_mamba=small
python -m spectramind train model.fgs1_mamba=medium
python -m spectramind train model.fgs1_mamba=large