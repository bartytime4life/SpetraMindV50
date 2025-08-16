# SpectraMind V50 — Model Hydra Group

This folder contains the Hydra config group for assembling the SpectraMind V50 model stack.

## Quick Start
Use the master group:
- `model=model` to load default components:
  - `encoder_fgs1: fgs1_mamba`
  - `encoder_airs: airs_gnn`
  - `decoder_mu: multi_scale_decoder`
  - `decoder_sigma: flow_uncertainty_head`
  - `corel: spectral_corel`

Override any subgroup at runtime, for example:
```

python -m spectramind.cli_core_v50 train model.decoder_mu=moe_decoder

```

Or override parameters:
```

python -m spectramind.cli_core_v50 train model.encoder_airs.hidden_dim=192

```

The Python factory at `src/spectramind/models/factory.py` consumes `cfg.model` and returns a
`BuiltModels` container with ready-to-use modules for training/inference.
