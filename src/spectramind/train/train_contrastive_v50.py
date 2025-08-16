from __future__ import annotations

"""
SpectraMind V50 — Contrastive Pretraining (Latent Alignment)

Features:

* InfoNCE-style loss with temperature
* Multi-view augment hooks (delegated to dataset)
* AMP, grad accumulation, cosine LR warmup
* JSONL + MLflow logging, early stopping on val NCE loss
"""

from typing import Any, Dict

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
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    import hydra  # type: ignore
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    hydra = None
    OmegaConf = None  # type: ignore


def _build_model(cfg: Dict[str, Any]):
    mod, fn = cfg["model"]["module"].split(":")
    return getattr(importlib.import_module(mod), fn)(cfg)  # type: ignore


def _infonce(z1, z2, temp: float = 0.1):
    """Standard InfoNCE loss for paired views z1, z2."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def run_contrastive(cfg: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_logger()
    set_seed(int(cfg.get("seed", 42)))
    meta = capture_run_meta(
        cfg, cli_version=str(cfg.get("cli_version", "v50")), seed=int(cfg.get("seed", 42))
    )
    write_debug_log(meta)
    try_mlflow_start(
        run_name=f"contrastive_v50_{meta.config_hash}",
        tags={"cfg": meta.config_hash, "ver": meta.cli_version},
    )
    try_mlflow_log_params({"cfg_hash": meta.config_hash, "cli_version": meta.cli_version})

    if torch is None:
        raise RuntimeError("PyTorch is required for contrastive pretraining.")

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
    ckpt_dir = cfg["trainer"].get("ckpt_dir", "artifacts/contrastive_checkpoints")
    checkpointer = Checkpointer(
        ckpt_dir, keep_last=int(cfg["trainer"].get("keep_last", 5))
    )
    early = EarlyStopper(
        patience=int(cfg["trainer"]["early_stop"].get("patience", 8)),
        min_delta=float(cfg["trainer"]["early_stop"].get("min_delta", 0.0)),
        metric_name="val_loss",
    )

    amp = bool(cfg["trainer"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    accum = int(cfg["trainer"].get("grad_accum", 1))
    log_every = int(cfg["trainer"].get("log_every", 50))
    temp = float(cfg.get("contrastive", {}).get("temperature", 0.1))

    global_step = 0
    best = None
    for epoch in range(1, int(cfg["trainer"]["max_epochs"]) + 1):
        model.train()
        running: Dict[str, float] = {}
        for it, batch in enumerate(train_loader, start=1):
            # Expect dataset to return two views: batch["view1"], batch["view2"]
            view1, view2 = batch["view1"].to(device, non_blocking=True), batch["view2"].to(
                device, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
                enabled=amp,
            ):
                z1, z2 = model(view1), model(view2)
                loss = _infonce(z1, z2, temp=temp) / accum
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
                    write_jsonl({"event": "contrastive_train_step", "step": global_step, **avg})
                    try_mlflow_log_metrics(avg, step=global_step)
                    running = {}

        # Validation loop (paired views)
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                v1, v2 = batch["view1"].to(device, non_blocking=True), batch["view2"].to(
                    device, non_blocking=True
                )
                z1, z2 = model(v1), model(v2)
                vals.append(float(_infonce(z1, z2, temp=temp).item()))
        vmean = float(sum(vals) / max(1, len(vals)))
        write_jsonl({"event": "contrastive_val_epoch", "epoch": epoch, "val_loss": vmean})
        try_mlflow_log_metrics({"val_loss": vmean}, step=epoch)

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
    export_manifest(manifest, out_path="artifacts/contrastive_manifest.json")
    return manifest


if hydra is not None:

    @hydra.main(config_path=None, config_name=None, version_base=None)
    def _main(cfg):  # type: ignore
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        run_contrastive(cfg_dict)  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    if hydra is None:
        raise SystemExit("Hydra is required to run this module as a script.")
    _main()  # type: ignore

