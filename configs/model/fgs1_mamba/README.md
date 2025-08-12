# FGS1 Mamba SSM — Encoder Config Group

Purpose: Modular configs for the FGS1 (temporal) encoder used by SpectraMind V50. These override only
`model.encoders.fgs1_mamba` fields, keeping the rest of the model untouched.

## How to use

From the project root:

- Compose against your main model config:
  ```bash
  python -m spectramind train model=base +model.encoders.fgs1_mamba=@fgs1_mamba/base