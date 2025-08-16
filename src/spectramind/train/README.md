# SpectraMind V50 — `src/spectramind/train`

Mission-grade training package for the NeurIPS 2025 Ariel Data Challenge (V50 stack).

## Features

* Deterministic seeding and environment capture
* Console + rotating file logs and a JSONL event stream
* Optional MLflow sync (`run.mlflow: true`)
* Hydra/OmegaConf-friendly config dicts; no hard dependency on Hydra to execute
* Flexible registry builders for datasets, models, losses, optimizers, schedulers, and step processors
* Robust `TrainerBase` with AMP, gradient accumulation, checkpointing, and early stopping
* Built-in dummy dataset/model for smoke testing
* `selftest.py` to validate the package in isolation

## Quick Start (Dummy Smoke Test)

```bash
python -m src.spectramind.train.train_v50
# or
python -m src.spectramind.train.selftest
```

Artifacts appear under `runs/...` including `train.log` (rotating text log) and `events.jsonl` (JSONL event stream).

## Integrating with the CLI

The `run_train(cfg: Dict[str, Any])` functions are designed to be called from SpectraMind’s Typer CLI modules:

* General supervised: `train_v50.run_train(cfg)`
* MAE pretrain: `train_mae_v50.run_train(cfg)`
* Contrastive pretrain: `train_contrastive_v50.run_train(cfg)`

## Config Schema (Example)

See the inline example at the top of `train_v50.py`. Minimal fields:

* `run`: name, out_dir, seed, device, mlflow flags
* `data`: dataset targets for train/val + dataloader options
* `model`: target + params
* `loss`: main + optional smooth/asym with weights
* `optim`: name/lr/weight_decay
* `sched`: scheduler config (cosine_warmup/onecycle/step/none)
* `train`: epochs, grad_accum, amp, clip_grad_norm, patience

## Reproducibility

* Config snapshots saved to `<out_dir>/config_snapshot.yaml`
* Git hash captured when available
* Logs and JSONL stream suitable for CI parsing
