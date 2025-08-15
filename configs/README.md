# SpectraMind V50 — Configs

Hydra configuration hub for the SpectraMind V50 pipeline. Import
`config_v50.yaml` as the root and override components at the CLI, e.g.:

```bash
python spectramind.py train +data=local +train=supervised +model=fgs1_mamba
```

### Layout

- `data/`         — Paths, calibration flags, platform overrides
- `model/`        — Encoder/decoder settings and overrides
- `train/`        — Phase configs (MAE pretrain, contrastive, supervised)
- `diagnostics/`  — Dashboard, UMAP/t-SNE, GLL heatmaps, leaderboard
- `symbolic/`     — Molecule bands, profiles, and override packs

Logging defaults capture console output, a rotating file log, and a JSONL
event stream. Experiment tracking toggles are exposed for MLflow and
Weights & Biases. Reproducibility helpers snapshot git metadata and the
runtime environment.

> Platform overrides should only adjust values under `paths`, `io`, or
> `loader`. Keep configs declarative and composable.

