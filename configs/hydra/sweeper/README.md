# Hydra Sweeper Configs

This directory defines hyperparameter search strategies for **SpectraMind V50**.

## Files
- **optuna.yaml** — Optuna-based hyperparameter sweeper (default).
  - Uses TPE sampler with deterministic seed.
  - Supports `training.lr`, `batch_size`, `encoder_dim`, `dropout`, and `weight_decay`.
  - Objective: minimize Generalized Log Likelihood (GLL).
- **basic.yaml** — Minimal Hydra sweeper (grid/random) for debugging.

## Usage

### Run Optuna sweep (default)
```bash
python train_v50.py -m
```

### Override sweeper

```bash
python train_v50.py -m hydra/sweeper=basic
```

### Override search space at CLI

```bash
python train_v50.py -m training.lr=tag(log,interval(1e-4,1e-2)) model.dropout=interval(0.1,0.3)
```

## Notes

* All sweeps log metadata (commit hash, ENV, config hash) for reproducibility.
* Parallelism (`n_jobs`) integrates with Hydra launcher configs (see `configs/hydra/launcher`).
* For large sweeps, configure persistent Optuna storage (e.g., SQLite or PostgreSQL).
