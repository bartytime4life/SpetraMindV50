"""
SpectraMind V50 - Training Package
==================================

This package provides mission-grade training utilities for the NeurIPS Ariel Data Challenge 2025:

* Deterministic seeding and environment capture
* Console + rotating file logs AND JSONL event stream
* Optional MLflow/W&B sync (MLflow integrated; W&B can be added similarly)
* Hydra/OmegaConf-friendly builders (datasets, models, losses, optimizers, schedulers, step processors)
* Robust TrainerBase with AMP, grad accumulation, checkpointing, early stopping, and reproducibility
* Self-test harness for quick validation of the stack

Primary public entry points:

* train_v50.run_train(cfg)  -> general supervised training
* train_mae_v50.run_train(cfg)  -> masked autoencoder pretraining
* train_contrastive_v50.run_train(cfg) -> contrastive pretraining

These functions are designed to be called from CLI modules (e.g., spectramind CLI).
"""

from .utils import seed_everything, get_device, count_parameters, dump_yaml_safely
from .experiment_logger import ExperimentLogger
from .trainer_base import TrainerBase, TrainerHooks
from .callbacks import EarlyStopping, CheckpointManager
from .optim import build_optimizer, build_scheduler
from .losses import GaussianLikelihoodLoss, SmoothnessLoss, AsymmetryLoss, CompositeLoss
from .registry import build_from_target, resolve_callable
from .step_processors import GenericGaussianProcessor

__all__ = [
    "seed_everything", "get_device", "count_parameters", "dump_yaml_safely",
    "ExperimentLogger",
    "TrainerBase", "TrainerHooks",
    "EarlyStopping", "CheckpointManager",
    "build_optimizer", "build_scheduler",
    "GaussianLikelihoodLoss", "SmoothnessLoss", "AsymmetryLoss", "CompositeLoss",
    "build_from_target", "resolve_callable",
    "GenericGaussianProcessor",
]
