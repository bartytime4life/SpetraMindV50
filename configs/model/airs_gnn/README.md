# AIRS GNN Config Set (Hydra-ready)

This directory contains Hydra component configs for the **AIRS spectral graph encoder** used by SpectraMind V50. They’re designed to be composed under the root `config_v50.yaml` via `model.encoders.airs_gnn`.

Highlights:
- Physics-informed graph: wavelength adjacency, molecule windows, and detector seams centralized in `physics_edges.yaml`.
- Edge features: \u0394\u03bb, molecule tags, seam flags, with optional learned/Fourier encodings.
- Backends: GAT (default), plus RGCN and NNConv flavors.
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
| tiny   | 96         | 3        | 3     | 0.10    |
| small  | 160        | 4        | 4     | 0.10    |
| medium | 224        | 5        | 4     | 0.10    |
| large  | 320        | 7        | 6     | 0.10    |

