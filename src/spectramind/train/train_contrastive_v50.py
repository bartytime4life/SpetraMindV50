"""
Contrastive pretraining runner for SpectraMind V50.

Supports generic contrastive losses (InfoNCE-style) via a model that returns embeddings for two
augmentations or views of the same input, and a loss module provided via config.

Expected step processor behavior:

* batch provides two views: x1, x2 (and optional labels/meta)
* model(x1, x2) -> dict with 'z1', 'z2' (embeddings)
* loss_module(z1, z2) -> dict with {'total': loss, 'nce': ..., ...}
"""
from typing import Any, Dict
import math
import os

import torch
from torch.utils.data import DataLoader

from .callbacks import CheckpointManager, EarlyStopping
from .experiment_logger import ExperimentLogger
from .optim import build_optimizer, build_scheduler
from .registry import build_from_target
from .trainer_base import TrainerBase
from .utils import dump_yaml_safely, ensure_dir, get_device, seed_everything


class _ContrastiveProcessor:
    def __init__(self, loss_module: torch.nn.Module) -> None:
        self.loss_module = loss_module

    def __call__(self, model: torch.nn.Module, batch: Dict, device: torch.device):
        x1 = batch["x1"].to(device)
        x2 = batch["x2"].to(device)
        out = model(x1, x2)  # expected to provide 'z1', 'z2'
        z1, z2 = out["z1"], out["z2"]
        losses = self.loss_module(z1, z2)
        total = losses["total"]
        scalars = {k: float(v.detach().item()) for k, v in losses.items()}
        return total, scalars


def run_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run = cfg.get("run", {})
    out_dir = run.get("out_dir", "runs/contrastive")
    run_name = run.get("name", "v50-contrastive")
    seed = int(run.get("seed", 1337))
    device_pref = run.get("device", "cuda")

    ensure_dir(out_dir)
    seed_everything(seed)
    logger = ExperimentLogger(run_name=run_name, log_dir=out_dir, mlflow_enable=bool(run.get("mlflow", False)), mlflow_experiment=run.get("mlflow_experiment"))
    logger.log_params({"cfg": cfg})

    device = get_device(device_pref)
    logger.info(f"Using device: {device}")

    # Data
    data_cfg = cfg.get("data", {})
    train_ds = build_from_target(data_cfg["train"])
    val_ds = build_from_target(data_cfg["val"]) if "val" in data_cfg and data_cfg["val"] else None
    loader_cfg = data_cfg.get("loader", {"batch_size": 64, "num_workers": 0, "pin_memory": False})
    train_loader = DataLoader(train_ds, batch_size=int(loader_cfg.get("batch_size", 64)), shuffle=True, num_workers=int(loader_cfg.get("num_workers", 0)), pin_memory=bool(loader_cfg.get("pin_memory", False)), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(loader_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(loader_cfg.get("num_workers", 0)), pin_memory=bool(loader_cfg.get("pin_memory", False)), drop_last=False) if val_ds is not None else None

    # Model & Loss
    model = build_from_target(cfg["model"])
    loss_module = build_from_target(cfg["loss"])  # should return dict with 'total'
    step_proc = _ContrastiveProcessor(loss_module)

    # Optim/Sched
    optim_cfg = cfg.get("optim", {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01})
    optimizer = build_optimizer(model, optim_cfg)
    total_steps = int(cfg.get("train", {}).get("epochs", 20)) * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg.get("sched"), total_steps=total_steps)

    # Trainer
    ckpt = CheckpointManager(out_dir)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=int(cfg.get("train", {}).get("patience", 5)))
    trainer = TrainerBase(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_proc,
        device=get_device(device_pref),
        logger=logger,
        grad_accum_steps=int(cfg.get("train", {}).get("grad_accum", 1)),
        use_amp=bool(cfg.get("train", {}).get("amp", True)),
        clip_grad_norm=float(cfg.get("train", {}).get("clip_grad_norm", 1.0)),
        checkpoint=ckpt,
        early_stopping=early if val_loader is not None else None,
    )

    dump_yaml_safely(cfg, os.path.join(out_dir, "config_snapshot.yaml"))

    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg.get("train", {}).get("epochs", 20)),
        start_epoch=1,
    )
    logger.info(f"Contrastive pretraining completed. Best val loss: {summary['best_val_loss']:.6f}")
    logger.close()
    return summary


if __name__ == "__main__":
    import sys
    print("Run this via the SpectraMind CLI or call run_train(cfg) with contrastive dataset+model+loss.")
    sys.exit(0)
