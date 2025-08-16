from __future__ import annotations

"""
SpectraMind V50 — Supervised Training (μ/σ) with Physics/Symbolic Losses

Hydra config expectations (example):
trainer:
  max_epochs: 50
  grad_accum: 1
  amp: true
  clip_grad: 1.0
  log_every: 50
  ckpt_dir: "artifacts/checkpoints"
  save_every_epochs: 1
  early_stop:
    patience: 7
    min_delta: 0.0
model:
  module: "spectramind.models.v50_model:build"
optimizer:
  lr: 3e-4
  wd: 1e-4
  betas: [0.9, 0.999]
scheduler:
  type: "cosine"
  warmup_steps: 500
  total_steps: 10000
loss:
  gll: { weight: 1.0 }
  smooth: { weight: 0.01 }
  fft: { weight: 0.0, hi_freq_cut: null }
  symbolic: { weight: 0.0, callable: "" }
data:
  module: "spectramind.data.ariel_dataset:build"
  batch_size: 16
  num_workers: 4
  pin_memory: true
seed: 42
"""

import os
from typing import Any, Dict, Tuple

import numpy as np

from .callbacks import Checkpointer, EarlyStopper
from .common import capture_run_meta, get_device, set_seed, write_debug_log
from .data_loading import build_dataloaders
from .logger import (
    get_logger,
    try_mlflow_end,
    try_mlflow_log_metrics,
    try_mlflow_log_params,
    try_mlflow_start,
    write_jsonl,
)
from .losses import assemble_losses
from .schedulers import cosine_with_warmup

import importlib

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import hydra  # type: ignore
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    hydra = None
    OmegaConf = None  # type: ignore


def _build_model(cfg: Dict[str, Any]) -> "torch.nn.Module":
    """Dynamically build the model from a registry string."""
    mod_name, fn_name = cfg["model"]["module"].split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn(cfg)  # type: ignore


def _build_optimizer(model: "torch.nn.Module", cfg: Dict[str, Any]) -> "torch.optim.Optimizer":
    """AdamW optimizer with Hydra-provided hyperparameters."""
    opt_cfg = cfg["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-4)),
        weight_decay=float(opt_cfg.get("wd", 1e-4)),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
    )


def _forward_step(model, batch) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Forward pass wrapper. Expect the model to return (mu, sigma) given inputs."""
    # Example batch contract:
    # batch = {"x": tensor[B,...], "target": tensor[B, D]}
    x = batch["x"]
    target = batch["target"]
    mu, sigma = model(x)
    return mu, sigma, target


def _evaluate(model, loader, cfg) -> Dict[str, float]:
    """Eval loop on a dataloader split."""
    model.eval()
    metrics_accum = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = (
                            v.cuda(non_blocking=True)
                            if next(model.parameters()).is_cuda
                            else v
                        )
            mu, sigma, target = _forward_step(model, batch)
            loss, parts = assemble_losses(mu, sigma, target, cfg["loss"])
            val = float(loss.item() if torch.is_tensor(loss) else loss)
            parts = {f"val_{k}": float(v) for k, v in parts.items()}
            parts["val_loss"] = val
            metrics_accum.append(parts)
    # Average
    if not metrics_accum:
        return {"val_loss": float("nan")}
    keys = metrics_accum[0].keys()
    out = {k: float(np.mean([m[k] for m in metrics_accum if k in m])) for k in keys}
    return out


def _maybe_amp(cfg: Dict[str, Any]):
    """Return a torch.cuda.amp autocast context manager if AMP enabled."""
    if not cfg["trainer"].get("amp", True):
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16
    )


def _grad_clip(model: "torch.nn.Module", max_norm: float) -> None:
    """Clip gradients if ``max_norm`` > 0."""
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _total_steps(cfg: Dict[str, Any], train_len: int) -> int:
    """Estimate total optimizer steps for the scheduler."""
    epochs = int(cfg["trainer"]["max_epochs"])
    accum = int(cfg["trainer"].get("grad_accum", 1))
    return max(1, (train_len * epochs) // accum)


def _log_batch_metrics(step: int, metrics: Dict[str, float]) -> None:
    """Log metrics to JSONL & MLflow."""
    write_jsonl({"event": "train_step", "step": step, **metrics})
    try_mlflow_log_metrics(metrics, step=step)


def _log_val_metrics(epoch: int, metrics: Dict[str, float]) -> None:
    """Log validation metrics."""
    write_jsonl({"event": "val_epoch", "epoch": epoch, **metrics})
    try_mlflow_log_metrics(metrics, step=epoch)


def _log_params(params: Dict[str, Any]) -> None:
    """Log run params to MLflow if present."""
    try_mlflow_log_params(params)


def _move_batch_to_device(batch, model):
    """Move dict/tensor batch to same device as model's first parameter."""
    dev = next(model.parameters()).device
    if isinstance(batch, dict):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(dev, non_blocking=True)
    elif torch.is_tensor(batch):
        batch = batch.to(dev, non_blocking=True)
    return batch


def _train_one_epoch(model, loader, optimizer, scheduler, cfg, step0: int) -> Tuple[int, Dict[str, float]]:
    """One epoch training loop with grad accumulation and AMP."""
    model.train()
    accum = int(cfg["trainer"].get("grad_accum", 1))
    clip = float(cfg["trainer"].get("clip_grad", 0.0))
    log_every = int(cfg["trainer"].get("log_every", 50))
    running: Dict[str, float] = {}
    step = step0

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["trainer"].get("amp", True))

    for it, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, model)
        with _maybe_amp(cfg):
            mu, sigma, target = _forward_step(model, batch)
            loss, parts = assemble_losses(mu, sigma, target, cfg["loss"])
            if torch.is_tensor(loss):
                loss = loss / accum
        if torch.is_tensor(loss):
            scaler.scale(loss).backward()
        else:
            raise RuntimeError("Loss must be a torch tensor in supervised training.")

        if it % accum == 0:
            scaler.unscale_(optimizer)
            _grad_clip(model, clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            # Aggregate + log
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v)
            running["loss"] = running.get("loss", 0.0) + float(loss.item() * accum)
            step += 1
            if step % log_every == 0:
                avg = {k: v / log_every for k, v in running.items()}
                _log_batch_metrics(step, avg)
                running = {}
    # Return last step and last running (averaged if we ended mid-window)
    if running:
        denom = max(1, log_every)
        avg = {k: v / denom for k, v in running.items()}
    else:
        avg = {}
    return step, avg


def _build_scheduler(opt, cfg, train_len: int):
    """Construct scheduler per cfg.scheduler.type."""
    sch_cfg = cfg.get("scheduler", {}) or {}
    typ = (sch_cfg.get("type") or "").lower()
    if typ == "cosine":
        total = int(sch_cfg.get("total_steps", _total_steps(cfg, train_len)))
        warm = int(sch_cfg.get("warmup_steps", max(1, total // 20)))
        return cosine_with_warmup(opt, warmup_steps=warm, total_steps=total)
    elif typ == "none":
        return None
    else:
        return None


def _ensure_cuda_graph_compat(model):
    """Optional future: prepare CUDA graph capture. Currently noop for portability."""
    return


def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate the full supervised training run and return an artifact manifest."""
    logger = get_logger()
    set_seed(int(cfg.get("seed", 42)))
    meta = capture_run_meta(
        cfg, cli_version=str(cfg.get("cli_version", "v50")), seed=int(cfg.get("seed", 42))
    )
    write_debug_log(meta)
    try_mlflow_start(
        run_name=f"train_v50_{meta.config_hash}",
        tags={"cfg": meta.config_hash, "ver": meta.cli_version},
    )
    _log_params({"cfg_hash": meta.config_hash, "cli_version": meta.cli_version})

    # Device
    device, _ = get_device("cuda")
    logger.info(f"Using device: {device}")

    # Data
    dls = build_dataloaders(cfg)
    train_loader, val_loader = dls.train, dls.val

    # Model
    if torch is None:
        raise RuntimeError("PyTorch is required for training.")
    model = _build_model(cfg)
    model.to(device)
    _ensure_cuda_graph_compat(model)

    # Optim + Sched
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, train_len=len(train_loader))

    # Callbacks
    ckpt_dir = cfg["trainer"].get("ckpt_dir", "artifacts/checkpoints")
    checker = Checkpointer(
        ckpt_dir, keep_last=int(cfg["trainer"].get("keep_last", 5))
    )
    early = EarlyStopper(
        patience=int(cfg["trainer"]["early_stop"].get("patience", 7)),
        min_delta=float(cfg["trainer"]["early_stop"].get("min_delta", 0.0)),
        metric_name="val_loss",
    )

    # Loop
    global_step = 0
    best_path = None
    for epoch in range(1, int(cfg["trainer"]["max_epochs"]) + 1):
        logger.info(f"Epoch {epoch}/{cfg['trainer']['max_epochs']}")
        global_step, train_avg = _train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg, global_step
        )
        for k, v in (train_avg or {}).items():
            write_jsonl({"event": "train_epoch_avg", "epoch": epoch, k: float(v)})
        # Val
        val_metrics = _evaluate(model, val_loader, cfg)
        _log_val_metrics(epoch, val_metrics)
        # Checkpointing
        if epoch % int(cfg["trainer"].get("save_every_epochs", 1)) == 0:
            checker.save_periodic(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (
                        None
                        if scheduler is None
                        else scheduler.state_dict()
                        if hasattr(scheduler, "state_dict")
                        else {}
                    ),
                    "step": global_step,
                    "cfg": cfg,
                },
                tag=f"epoch_{epoch:03d}",
            )
        best_maybe = checker.try_save_best(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (
                    None
                    if scheduler is None
                    else scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else {}
                ),
                "step": global_step,
                "cfg": cfg,
            },
            value=float(val_metrics.get("val_loss", float("inf"))),
        )
        if best_maybe:
            best_path = best_maybe
        # Early stop
        if early.step(value=float(val_metrics.get("val_loss", float("inf")))):
            logger.info(
                f"Early stopping triggered at epoch {epoch} with best {early.best:.6f}."
            )
            break

    # Finalize
    try_mlflow_end()
    manifest = {
        "best_checkpoint": best_path,
        "cfg_hash": meta.config_hash,
        "events_jsonl": "train_events.jsonl",
        "log_file": "logs/train.log",
    }
    from .common import export_manifest

    export_manifest(manifest, out_path="artifacts/train_manifest.json")
    return manifest


if hydra is not None:

    @hydra.main(config_path=None, config_name=None, version_base=None)
    def _main(cfg):  # type: ignore
        """Hydra entrypoint. Accepts a composed config dictionary.

        Usage:
            python -m src.spectramind.train.train_v50 +trainer.max_epochs=2
        """
        # Convert OmegaConf to plain dict for hashing and logging
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        run_training(cfg_dict)  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    if hydra is None:
        raise SystemExit("Hydra is required to run this module as a script.")
    _main()  # type: ignore

