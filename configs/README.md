# SpectraMind V50 — Configurations (Hydra)

This directory contains all Hydra-compatible configuration files for the SpectraMind V50 neuro‑symbolic, physics‑informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge.

## Layout

- `hydra.yaml` — Global Hydra runtime/sweep directories & job naming.
- `defaults.yaml` — Single point of composition; includes the canonical base stacks.

### Trees

- `data/`         — Dataset paths, IO, feature-engineering toggles, platform overrides (local/kaggle/cluster).
- `model/`        — Encoders/decoders/fusion definitions (FGS1 Mamba, AIRS GNN, μ/σ decoders).
- `training/`     — Phase configs (base, MAE pretrain, contrastive, COREL).
- `calibration/`  — Post-hoc uncertainty calibration (temperature scaling + conformal/COREL).
- `diagnostics/`  — SHAP, symbolic, FFT, dashboards.
- `symbolic/`     — Physics/molecule/alignment rules & weights.

> Rule of Truth: platform overrides must only flip values under `paths`, `io`, or `loader`. Do not mutate structure across platforms. Keep configs declarative and composable.
