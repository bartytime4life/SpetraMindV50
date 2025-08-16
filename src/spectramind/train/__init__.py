from __future__ import annotations

"""
SpectraMind V50 — Training Package (src/spectramind/train)

This package provides unified, Hydra-safe, CLI-integrable training pipelines for the
NeurIPS 2025 Ariel Data Challenge (SpectraMind V50). It includes:

* train_v50.py:         Full μ/σ supervised training loop with physics-aware losses.
* train_mae_v50.py:     Masked Autoencoder (MAE) pretraining for spectral/time cubes.
* train_contrastive_v50.py: Contrastive pretraining on latent embeddings.
* corel_train.py:       COREL bin-wise conformal calibration training (GNN-ready).
* callbacks.py:         Early stopping, checkpointing, and graceful failover hooks.
* schedulers.py:        Cosine annealing with warmup; step and plateau schedulers.
* losses.py:            Gaussian log-likelihood (GLL), L2 smoothness, FFT penalties,
  symbolic wrapper hooks for physics-informed constraints.
* logger.py:            Console + rotating file logs + JSONL event stream + MLflow/W&B optional.
* common.py:            Deterministic seeding, device selection, checkpoint utils, hash capture.
* data_loading.py:      Thin data adapters and dataset registry plumbing.
* selftest_train_pkg.py: Coherence checks for this package (imports, shapes, I/O).

All modules write reproducibility metadata to `v50_debug_log.md` and support structured
JSONL logging for post-hoc analytics and dashboards.

Conventions:

* Do NOT import heavy frameworks at module import unless necessary. Delay where possible.
* Always provide rich docstrings and explicit typing.
* Every public function logs via the unified logger when meaningful.

Note:
This package intentionally keeps *no* project-specific absolute paths; it relies on Hydra
configs and the caller's working directory.
"""

__all__ = [
    "common",
    "logger",
    "losses",
    "schedulers",
    "callbacks",
    "data_loading",
]

