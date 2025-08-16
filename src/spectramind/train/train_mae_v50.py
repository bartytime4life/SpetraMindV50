from __future__ import annotations

"""
SpectraMind V50 — Masked Autoencoder (MAE) Pretraining

This module implements MAE-style pretraining for spectral/time cubes with support for:

* Random/block/molecule-region masking (delegated to dataset via cfg)
* AMP mixed precision
* Cosine LR with warmup
* Logging to JSONL and MLflow (optional)
* Checkpointing + Early stopping on reconstruction loss

Expected cfg sections:
mae:
  mask_ratio: 0.75
  loss: "l2"          # or "l1"
trainer: {...}        # same schema as supervised trainer where sensible
model:
  module: "spectramind.models.v50_mae:build"
data:
  module: "spectramind.data.ariel_mae_dataset_v50:build"
optimizer: {...}
scheduler: {...}
"""

from typing import Any, Dict, Tuple

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
from .schedulers import cosine_with_warmup

import importlib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:
    import hydra  # type: ignore
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    hydra = None
    OmegaConf = None  # type: ignore


def _build_model(cfg: Dict[str, Any]) -> "torch.nn.Module":
    mod_name, fn_name = cfg["model"]["module"].split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)(cfg)  # type: ignore


def _recon_loss(pred, target, typ: str = "l2"):
    if typ == "l1":
        return (pred - target).abs().mean()
    return ((pred - target) ** 2).mean()


def _forward_mae(model, batch):
    """
    Expect dataset to provide:
    batch["x"]: input tensor
    batch["mask"]: boolean mask of missing/hidden patches/bins
    model(x, mask) -> reconstruction over masked regions (or full with mask)
    batch["target"]: ground truth for reconstruction (same shape as model output)
    """
    x = batch["x"]
    mask = batch.get("mask", None)
    target = batch.get("target", x)
    if mask is None:
        out = model(x)
    else:
        out = model(x, mask=mask)
    return out, target


def run_mae_pretraining(cfg: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_logger()
    set_seed(int(cfg.get("seed", 42)))
    meta = capture_run_meta(
        cfg, cli_version=str(cfg.get("cli_version", "v50")), seed=int(cfg.get("seed", 42))
    )
    write_debug_log(meta)
    try_mlflow_start(
        run_name=f"mae_v50_{meta.config_hash}",
        tags={"cfg": meta.config_hash, "ver": meta.cli_version},
    )
    try_mlflow_log_params({"cfg_hash": meta.config_hash, "cli_version": meta.cli_version})

    if torch is None:
        raise RuntimeError("PyTorch is required for MAE pretraining.")

    device, _ = get_device("cuda")

    dls = build_dataloaders(cfg)
    train_loader, val_loader = dls.train, dls.val

    model = _build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"].get("lr", 3e-4)),
        weight_decay=float(cfg["optimizer"].get("wd", 1e-4)),
    )
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=int(cfg["scheduler"].get("warmup_steps", 500)),
        total_steps=int(cfg["scheduler"].get("total_steps", 10000)),
    )

    ckpt_dir = cfg["trainer"].get("ckpt_dir", "artifacts/mae_checkpoints")
    checkpointer = Checkpointer(
        ckpt_dir, keep_last=int(cfg["trainer"].get("keep_last", 5))
    )
    early = EarlyStopper(
        patience=int(cfg["trainer"]["early_stop"].get("patience", 10)),
        min_delta=float(cfg["trainer"]["early_stop"].get("min_delta", 0.0)),
        metric_name="val_loss",
    )

    amp = bool(cfg["trainer"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_typ = str(cfg["mae"].get("loss", "l2"))
    accum = int(cfg["trainer"].get("grad_accum", 1))
    log_every = int(cfg["trainer"].get("log_every", 50))

    global_step = 0
    best = None
    for epoch in range(1, int(cfg["trainer"]["max_epochs"]) + 1):
        model.train()
        running: Dict[str, float] = {}
        for it, batch in enumerate(train_loader, start=1):
            for k, v in (batch.items() if isinstance(batch, dict) else []):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
                enabled=amp,
            ):
                pred, tgt = _forward_mae(model, batch)
                loss = _recon_loss(pred, tgt, typ=loss_typ) / accum
            scaler.scale(loss).backward()
            if it % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                running["loss"] = running.get("loss", 0.0) + float(loss.item() * accum)
                if global_step % log_every == 0:
                    avg = {k: v / log_every for k, v in running.items()}
                    write_jsonl({"event": "mae_train_step", "step": global_step, **avg})
                    try_mlflow_log_metrics(avg, step=global_step)
                    running = {}

        # Validation
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                for k, v in (batch.items() if isinstance(batch, dict) else []):
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)
                pred, tgt = _forward_mae(model, batch)
                v = _recon_loss(pred, tgt, typ=loss_typ)
                vals.append(float(v.item()))
        vmean = float(sum(vals) / max(1, len(vals)))
        write_jsonl({"event": "mae_val_epoch", "epoch": epoch, "val_loss": vmean})
        try_mlflow_log_metrics({"val_loss": vmean}, step=epoch)

        # Checkpointing
        checkpointer.save_periodic(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "cfg": cfg,
            },
            tag=f"epoch_{epoch:03d}",
        )
        maybe = checkpointer.try_save_best(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "cfg": cfg,
            },
            value=vmean,
        )
        if maybe:
            best = maybe

        if early.step(vmean):
            get_logger().info(
                f"Early stopping at epoch {epoch} (best {early.best:.6f})."
            )
            break

    try_mlflow_end()
    from .common import export_manifest

    manifest = {
        "best_checkpoint": best,
        "events_jsonl": "train_events.jsonl",
        "log_file": "logs/train.log",
        "cfg_hash": meta.config_hash,
    }
    export_manifest(manifest, out_path="artifacts/mae_manifest.json")
    return manifest


if hydra is not None:

    @hydra.main(config_path=None, config_name=None, version_base=None)
    def _main(cfg):  # type: ignore
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        run_mae_pretraining(cfg_dict)  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    if hydra is None:
        raise SystemExit("Hydra is required to run this module as a script.")
    _main()  # type: ignore

