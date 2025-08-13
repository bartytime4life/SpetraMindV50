# AIRS GNN Config Set (Hydra-ready)

This directory contains Hydra component configs for the **AIRS spectral graph encoder** used by SpectraMind V50.
They’re designed to be composed under the root `config_v50.yaml` via `model.encoders.airs_gnn`.

Highlights:
- Physics-informed graph: edges from wavelength adjacency, molecule bands, and detector seams.
- Edge features: Δλ, molecule tags, seam flags; optional learnable encodings.
- Attention-friendly backends: GAT (default), plus RGCN and NNConv flavors.
- TorchScript/JIT-safe model contracts, dropout, residuals, and attention export toggles.
- Size presets: `tiny`, `small`, `medium`, `large` for quick runtime control (Kaggle ≤9h guardrail).

## How to use (examples)

Default (GAT, medium):

```bash
python -m spectramind train model.airs_gnn=medium
```

RGCN variant:

```bash
python -m spectramind train model.airs_gnn=medium model.airs_gnn@model.encoders.airs_gnn.backend=rgcn
```

Custom preset override:

```bash
python -m spectramind train model.airs_gnn=small model.encoders.airs_gnn.hidden_dim=256 model.encoders.airs_gnn.n_layers=4
```

## Size presets

| preset | hidden_dim | n_layers | heads | dropout |
|--------|------------|----------|-------|---------|
| small  | 128        | 2        | 2     | 0.05    |
| medium | 192        | 3        | 4     | 0.10    |

*Tiny and large presets will be documented when available.*
