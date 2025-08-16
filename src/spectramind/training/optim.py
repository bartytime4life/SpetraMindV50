from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch.optim import Optimizer


def build_optimizer(params, cfg: Dict[str, Any]) -> Optimizer:
    """
    Factory for optimizers.
    cfg example:
    {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999], "momentum": 0.9}
    """
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))

    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", True))
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer: Optimizer, cfg: Dict[str, Any], steps_per_epoch: int) -> Tuple[Any, str]:
    """
    Factory for LR schedulers. Returns (scheduler, step_mode)
    step_mode in {"epoch", "step"} for when to call scheduler.step().
    cfg example:
    {"name": "cosine", "epochs": 50, "warmup_epochs": 5, "min_lr": 1e-6}
    """
    name = str(cfg.get("name", "cosine")).lower()
    if name == "none":
        return None, "epoch"

    if name == "cosine":
        epochs = int(cfg.get("epochs", 50))
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        min_lr = float(cfg.get("min_lr", 1e-6))
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            scale = (min_lr / optimizer.defaults["lr"]) + (1 - min_lr / optimizer.defaults["lr"]) * cosine
            return scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, "step"

    if name == "multistep":
        milestones = list(cfg.get("milestones", [30, 40]))
        gamma = float(cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return scheduler, "epoch"

    if name == "linear":
        epochs = int(cfg.get("epochs", 50))
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, "step"

    raise ValueError(f"Unknown scheduler '{name}'")
