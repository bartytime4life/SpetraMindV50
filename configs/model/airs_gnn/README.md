# /configs/model/airs_gnn

Hydra YAML configs for the **AIRS physics‑informed GNN encoder** (283 spectral bins as nodes).
These override only `model.encoders.airs_gnn` so you can tune the spectral graph stack without
touching fusion, decoders, or FGS1.

## Usage

Override just the AIRS encoder from the CLI:
```bash
python -m spectramind train model.airs_gnn=base
python -m spectramind train model.airs_gnn=small
python -m spectramind train model.airs_gnn=medium
python -m spectramind train model.airs_gnn=large
# Optional: swap edge/PE/explainability options
python -m spectramind train model.airs_gnn=base model.airs_gnn@model.encoders.airs_gnn.edges=edges