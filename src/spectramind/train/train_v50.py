"""
train_v50.py
=============

General supervised training entry point for SpectraMind V50.

Design:

* Hydra/OmegaConf friendly (but does not hard-depend on Hydra to remain runnable)
* Uses registry-style builders for datasets, model, loss, optimizer, scheduler, and step processor
* Mission-grade logging via ExperimentLogger (console + rotating file + JSONL + optional MLflow)
* Deterministic seeding, AMP, grad accumulation, checkpointing, early stopping

Example (with OmegaConf YAML):
cfg:
  run:
    name: "v50-trial"
    out_dir: "runs/v50-trial"
    seed: 1337
    device: "cuda"
    mlflow: false
    mlflow_experiment: null
  data:
    train:
      target: "your_pkg.data:YourTrainDataset"
      params: { ... }
    val:
      target: "your_pkg.data:YourValDataset"
      params: { ... }
    loader:
      batch_size: 16
      num_workers: 4
      pin_memory: true
  model:
    target: "your_pkg.models:YourModel"
    params: { ... }
  loss:
    main:
      target: "spectramind.train.losses:GaussianLikelihoodLoss"
      params: { "min_sigma": 1e-3 }
    smooth:
      target: "spectramind.train.losses:SmoothnessLoss"
      params: { "weight": 0.1 }
    asym:
      target: "spectramind.train.losses:AsymmetryLoss"
      params: { "weight": 0.05 }
    weights:
      lambda_smooth: 0.1
      lambda_asym: 0.05
  step_processor:
    target: "spectramind.train.step_processors:GenericGaussianProcessor"
    params: {}
  optim:
    name: "adamw"
    lr: 0.0003
    weight_decay: 0.01
  sched:
    name: "cosine_warmup"
    warmup_steps: 500
  train:
    epochs: 20
    grad_accum: 1
    amp: true
    clip_grad_norm: 1.0
    patience: 8

A minimal built-in dummy mode is provided for smoke testing if no model/dataset are configured.
"""
import math
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .callbacks import CheckpointManager, EarlyStopping
from .experiment_logger import ExperimentLogger
from .losses import AsymmetryLoss, CompositeLoss, GaussianLikelihoodLoss, SmoothnessLoss
from .optim import build_optimizer, build_scheduler
from .registry import build_from_target
from .step_processors import GenericGaussianProcessor
from .trainer_base import TrainerBase, TrainerHooks
from .utils import RunContext, dump_yaml_safely, ensure_dir, get_device, seed_everything

# -------------------------

# Built-in dummy components

# -------------------------

class _DummyDataset(Dataset):
    def __init__(self, n: int = 1024, dims: int = 64, noise: float = 0.1):
        self.x = torch.randn(n, dims)
        w = torch.randn(dims, 32) * 0.1
        mu = self.x @ w
        self.y = mu + noise * torch.randn_like(mu)
        self.dims = dims

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class _DummyModel(torch.nn.Module):
    def __init__(self, inp: int = 64, out: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self.mu_head = torch.nn.Linear(64, out)
        self.sigma_head = torch.nn.Linear(64, out)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h)
        return {"mu": mu, "sigma": sigma}


def _make_dummy(cfg: Dict[str, Any]):
    dims = int(cfg.get("dummy_dims", 64))
    out_bins = int(cfg.get("dummy_bins", 32))
    train_ds = _DummyDataset(n=1024, dims=dims)
    val_ds = _DummyDataset(n=256, dims=dims)
    model = _DummyModel(inp=dims, out=out_bins)
    return train_ds, val_ds, model

# -------------------------

# Runner

# -------------------------

def _build_loss(cfg_loss: Dict[str, Any]) -> CompositeLoss:
    main_cfg = cfg_loss.get("main") or {"target": "spectramind.train.losses:GaussianLikelihoodLoss", "params": {}}
    main = build_from_target(main_cfg)
    smooth = None
    asym = None
    if "smooth" in cfg_loss and cfg_loss["smooth"]:
        smooth = build_from_target(cfg_loss["smooth"])
    if "asym" in cfg_loss and cfg_loss["asym"]:
        asym = build_from_target(cfg_loss["asym"])
    weights = cfg_loss.get("weights") or {}
    return CompositeLoss(main, smooth, asym, float(weights.get("lambda_smooth", 0.0)), float(weights.get("lambda_asym", 0.0)))

def _dataloader(ds: Dataset, loader_cfg: Dict[str, Any]) -> DataLoader:
    bs = int(loader_cfg.get("batch_size", 16))
    nw = int(loader_cfg.get("num_workers", 0))
    pm = bool(loader_cfg.get("pin_memory", False))
    shuffle = bool(loader_cfg.get("shuffle", True))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pm, drop_last=True)


def run_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute supervised training given a Hydra/OmegaConf-like dict.

    Returns: summary dict with key 'best_val_loss'.
    """
    # --- Run setup ---
    run = cfg.get("run", {})
    out_dir = run.get("out_dir", "runs/v50")
    run_name = run.get("name", "v50-train")
    seed = int(run.get("seed", 1337))
    device_pref = run.get("device", "cuda")
    mlflow_enable = bool(run.get("mlflow", False))
    mlflow_experiment = run.get("mlflow_experiment")

    ensure_dir(out_dir)
    seed_everything(seed)

    logger = ExperimentLogger(run_name=run_name, log_dir=out_dir, mlflow_enable=mlflow_enable, mlflow_experiment=mlflow_experiment)
    logger.log_params({"cfg": cfg})

    device = get_device(device_pref)
    logger.info(f"Using device: {device}")

    # --- Data & Model ---
    if cfg.get("dummy_mode", False):
        train_ds, val_ds, model = _make_dummy(cfg.get("dummy", {}))
        loader_cfg = {"batch_size": 64, "num_workers": 0, "pin_memory": False}
    else:
        data_cfg = cfg.get("data", {})
        train_ds = build_from_target(data_cfg["train"])
        val_ds = build_from_target(data_cfg["val"]) if "val" in data_cfg and data_cfg["val"] else None
        loader_cfg = data_cfg.get("loader", {"batch_size": 16, "num_workers": 0, "pin_memory": False})

        model_cfg = cfg.get("model")
        if not model_cfg:
            raise ValueError("Missing 'model' config block.")
        model = build_from_target(model_cfg)

    train_loader = _dataloader(train_ds, loader_cfg)
    val_loader = _dataloader(val_ds, {**loader_cfg, "shuffle": False}) if val_ds is not None else None

    # --- Loss & Step Processor ---
    loss = _build_loss(cfg.get("loss", {}))
    step_proc_cfg = cfg.get("step_processor", {"target": "spectramind.train.step_processors:GenericGaussianProcessor", "params": {}})
    step_proc = build_from_target({**step_proc_cfg, "params": {**(step_proc_cfg.get("params") or {}), "loss_module": loss}})

    # --- Optimizer & Scheduler ---
    optim_cfg = cfg.get("optim", {"name": "adamw", "lr": 3e-4, "weight_decay": 0.01})
    optimizer = build_optimizer(model, optim_cfg)

    total_steps = math.ceil(len(train_loader) / max(1, int(cfg.get("train", {}).get("grad_accum", 1)))) * int(cfg.get("train", {}).get("epochs", 10))
    sched = build_scheduler(optimizer, cfg.get("sched"), total_steps=total_steps)

    # --- Trainer ---
    train_cfg = cfg.get("train", {})
    ckpt = CheckpointManager(out_dir)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=int(train_cfg.get("patience", 10)))

    trainer = TrainerBase(
        model=model,
        optimizer=optimizer,
        scheduler=sched,
        step_fn=step_proc,
        device=device,
        logger=logger,
        grad_accum_steps=int(train_cfg.get("grad_accum", 1)),
        use_amp=bool(train_cfg.get("amp", True)),
        clip_grad_norm=float(train_cfg.get("clip_grad_norm", 1.0)) if train_cfg.get("clip_grad_norm", None) is not None else None,
        checkpoint=ckpt,
        early_stopping=early if val_loader is not None else None,
        hooks=TrainerHooks(),
    )

    # Snapshot config for reproducibility
    dump_yaml_safely(cfg, os.path.join(out_dir, "config_snapshot.yaml"))

    # --- Train ---
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_cfg.get("epochs", 10)),
        start_epoch=1,
    )

    logger.info(f"Training completed. Best val loss: {summary['best_val_loss']:.6f}")
    logger.close()
    return summary


if __name__ == "__main__":
    # Minimal CLI to run dummy smoke test if no external CLI provided.
    # Example:
    #   python -m src.spectramind.train.train_v50
    cfg = {
        "run": {"name": "v50-dummy", "out_dir": "runs/v50-dummy", "seed": 1337, "device": "cpu", "mlflow": False},
        "dummy_mode": True,
        "dummy": {"dummy_dims": 64, "dummy_bins": 32},
        "loss": {
            "main": {"target": "spectramind.train.losses:GaussianLikelihoodLoss", "params": {"min_sigma": 1e-3}},
            "weights": {"lambda_smooth": 0.0, "lambda_asym": 0.0},
        },
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01},
        "sched": {"name": "cosine_warmup", "warmup_steps": 50},
        "train": {"epochs": 3, "grad_accum": 1, "amp": False, "clip_grad_norm": 1.0, "patience": 3},
    }
    run_train(cfg)
