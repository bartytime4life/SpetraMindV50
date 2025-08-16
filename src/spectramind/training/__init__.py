"""
SpectraMind V50 Training Package

This package provides a mission-ready, production-grade training stack for the
NeurIPS Ariel Data Challenge 2025. It includes:

* Deterministic, Hydra-safe training orchestration
* AMP, grad accumulation, gradient clipping, EMA
* Checkpointing with best/last policy and resume support
* Rotating file logs + JSONL event stream + optional MLflow
* DDP-aware helpers (rank-zero safe logging)
* Config hashing, ENV capture, Git hash capture for reproducibility
* Core scientific losses (GLL, smoothness, asymmetry) and plug-in symbolic loss

Public API:
V50Trainer                : High-level trainer orchestrator
build_dataloaders         : DataLoader builders with seed workers
build_optimizer           : Optimizer factory (AdamW, SGD, etc)
build_scheduler           : Scheduler factory (cosine, linear, multistep)
compute_total_loss        : Composite training loss
compute_metrics           : Core metrics (GLL, RMSE, MAE, coverage)
"""

from .trainer import V50Trainer
from .dataset_utils import build_dataloaders
from .optim import build_optimizer, build_scheduler
from .losses import compute_total_loss
from .metrics import compute_metrics

__all__ = [
    "V50Trainer",
    "build_dataloaders",
    "build_optimizer",
    "build_scheduler",
    "compute_total_loss",
    "compute_metrics",
]
