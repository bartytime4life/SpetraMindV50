# SpectraMind V50 Training

Mission-ready training stack for the NeurIPS Ariel Data Challenge 2025.

**Highlights**

* Deterministic, Hydra-safe config usage
* AMP + grad accumulation + gradient clipping
* EMA, early stopping, best/last checkpoints
* Rotating logs + JSONL event stream + optional MLflow
* DDP-aware safe logging and barriers
* Core scientific losses (GLL, smoothness, asymmetry) and optional symbolic loss
* Minimal, stable APIs to plug into SpectraMind CLIs

**Quickstart (Pseudo)**

```python
from spectramind.training import V50Trainer
from spectramind.models import build_model  # your project builder
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "seed": 42,
    "run": {"dir": "runs", "mlflow": False},
    "data": {
        "train_builder": "spectramind.data.builders:make_train",
        "val_builder": "spectramind.data.builders:make_val",
        "batch_size": 16,
        "num_workers": 8,
    },
    "loss": {"gll": {"weight": 1.0}, "smooth": {"weight": 0.02}, "asym": {"weight": 0.01}},
    "training": {"epochs": 50, "amp": True, "grad_accum": 1, "max_grad_norm": 1.0, "monitor": "gll", "patience": 10},
    "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01},
    "scheduler": {"name": "cosine", "epochs": 50, "warmup_epochs": 3, "min_lr": 1e-6},
})
model = build_model(cfg)
trainer = V50Trainer(model, cfg)
best, last_val = trainer.fit()
```
