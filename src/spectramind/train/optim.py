from typing import Any, Dict, Optional
import math
import torch

def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer from config dict:
    cfg example:
    {
        "name": "adamw",
        "lr": 3e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8
    }
    """
    name = (cfg.get("name") or "adamw").lower()
    lr = float(cfg.get("lr", 3e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore
        eps = float(cfg.get("eps", 1e-8))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore
        eps = float(cfg.get("eps", 1e-8))
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", True))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Optional[Dict[str, Any]], total_steps: Optional[int] = None):
    """
    Build LR scheduler from config. Supported:
    - cosine: CosineAnnealingLR (requires T_max)
    - cosine_warmup: linear warmup to base LR then cosine decay over total_steps
    - onecycle: OneCycleLR (requires total_steps)
    - step: StepLR (step_size, gamma)

    Returns scheduler or None.
    """
    if not cfg:
        return None
    name = (cfg.get("name") or "none").lower()

    if name == "cosine":
        tmax = int(cfg.get("t_max", total_steps or 1000))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
    if name == "cosine_warmup":
        if total_steps is None:
            raise ValueError("cosine_warmup requires total_steps")
        warmup_steps = int(cfg.get("warmup_steps", max(1, int(0.05 * total_steps))))
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if name == "onecycle":
        if total_steps is None:
            raise ValueError("onecycle requires total_steps")
        max_lr = float(cfg.get("max_lr", 1e-3))
        pct_start = float(cfg.get("pct_start", 0.3))
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start)
    if name == "step":
        step_size = int(cfg.get("step_size", 1000))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name in ("none", "null"):
        return None
    raise ValueError(f"Unknown scheduler: {name}")
