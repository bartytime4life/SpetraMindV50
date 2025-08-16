from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .checkpoint import CheckpointManager
from .dataset_utils import build_dataloaders
from .distributed import barrier, init_distributed, is_main_process
from .early_stopping import EarlyStopping
from .ema import ExponentialMovingAverage
from .engine import TrainEngine
from .hooks import Callback, JsonlCallback, LRLoggerCallback, MlflowCallback, TimeMeterCallback
from .loggers import JSONLLogger, setup_logging
from .optim import build_optimizer, build_scheduler
from .profiler import get_profiler
from .utils import (
    config_hash,
    create_run_dir,
    env_info,
    ensure_dir,
    get_git_hash,
    rank_zero_only,
    seed_everything,
    stable_serialize_config,
    write_json,
)


class V50Trainer:
    """
    High-level training orchestrator for SpectraMind V50.

    Typical usage:
        trainer = V50Trainer(model, cfg)
        trainer.fit()

    Config keys (examples, not exhaustive):
      seed: 42
      run:
        dir: "runs"
        name: "v50"
        mlflow: false
      data:
        train_builder: "spectramind.data.builders:make_train"
        val_builder: "spectramind.data.builders:make_val"
        batch_size: 16
        num_workers: 8
      loss:
        gll: {weight: 1.0}
        smooth: {weight: 0.02}
        asym: {weight: 0.01}
        symbolic: {weight: 0.5}
      training:
        epochs: 50
        amp: true
        grad_accum: 1
        max_grad_norm: 1.0
        ema_decay: 0.999
        clip_value: 0.0  # optional
        monitor: "gll"   # metric to minimize
        patience: 10
      optimizer:
        name: "adamw"
        lr: 1e-3
        weight_decay: 0.01
      scheduler:
        name: "cosine"
        epochs: 50
        warmup_epochs: 3
        min_lr: 1.0e-6
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Dict[str, Any],
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        callbacks: Optional[list[Callback]] = None,
    ):
        self.model = model
        self.cfg = cfg
        self._init_seed()
        self._init_distributed()

        # Run directory and logging
        self.cfg_hash = config_hash(cfg)
        run_base = Path(cfg.get("run", {}).get("dir", "runs"))
        self.run_dir = create_run_dir(run_base, self.cfg_hash)
        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.log_dir = ensure_dir(self.run_dir / "logs")
        self.assets_dir = ensure_dir(self.run_dir / "assets")

        self.log_path = setup_logging(self.log_dir)
        self.jsonl_path = self.log_dir / "events.jsonl"
        self.jsonl = JSONLLogger(self.jsonl_path)

        # Save config snapshot and environment info
        self._write_run_metadata()

        # Data
        if train_loader is None or val_loader is None:
            self.train_loader, self.val_loader = build_dataloaders(self.cfg)
        else:
            self.train_loader, self.val_loader = train_loader, val_loader

        # Model / device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Optimizer / scheduler
        self.optimizer = build_optimizer(self.model.parameters(), self.cfg.get("optimizer", {}))
        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler, self.scheduler_mode = build_scheduler(
            self.optimizer, self.cfg.get("scheduler", {}), steps_per_epoch
        )

        # AMP scaler
        use_amp = bool(self.cfg.get("training", {}).get("amp", True)) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # EMA
        ema_decay = float(self.cfg.get("training", {}).get("ema_decay", 0.0))
        self.ema = ExponentialMovingAverage(self.model, decay=ema_decay) if ema_decay > 0 else None

        # Checkpointing
        self.ckpt = CheckpointManager(self.ckpt_dir)

        # Early stopping
        trn = self.cfg.get("training", {})
        self.monitor = str(trn.get("monitor", "gll"))
        self.early = EarlyStopping(
            patience=int(trn.get("patience", 10)),
            min_delta=float(trn.get("min_delta", 0.0)),
            mode="min",
        )

        # Callbacks
        self.callbacks: list[Callback] = callbacks or []
        self.callbacks += [
            JsonlCallback(self.jsonl),
            LRLoggerCallback(),
            TimeMeterCallback(),
            MlflowCallback(enabled=bool(self.cfg.get("run", {}).get("mlflow", False))),
        ]

        self.global_step = 0
        self.best_metric_value = float("inf")
        self.epoch = 0

    def _init_seed(self) -> None:
        seed_everything(int(self.cfg.get("seed", 42)))

    def _init_distributed(self) -> None:
        init_distributed()

    @rank_zero_only
    def _write_run_metadata(self) -> None:
        meta = {
            "cfg_hash": self.cfg_hash,
            "git_hash": get_git_hash(),
            "env": env_info(),
            "cfg": self.cfg,
        }
        write_json(self.run_dir / "run_meta.json", meta)
        # Also write a normalized config snapshot for reproducibility
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            f.write(stable_serialize_config(self.cfg))

    def _callbacks(self, method: str, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if fn is not None:
                fn(state)

    def _state(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optimizer": self.optimizer,
            "val_metrics": getattr(self, "last_val_metrics", {}),
            "run_meta": {"cfg_hash": self.cfg_hash, "git_hash": get_git_hash()},
            "final": {"best_metric_value": self.best_metric_value, "monitor": self.monitor},
        }

    def _maybe_scheduler_step(self, when: str) -> None:
        if self.scheduler is None:
            return
        if when == self.scheduler_mode:
            self.scheduler.step()

    def fit(self) -> Tuple[float, Dict[str, float]]:
        """
        Execute the full training loop. Returns best_metric_value and last val metrics.
        """
        self._callbacks("on_train_start", self._state())

        engine = TrainEngine(self.model, self.optimizer, self.device, self.scaler, self.cfg)
        epochs = int(self.cfg.get("training", {}).get("epochs", 50))

        with get_profiler(enabled=bool(self.cfg.get("training", {}).get("profile", False))):
            for self.epoch in range(1, epochs + 1):
                self._callbacks("on_epoch_start", self._state())
                self.global_step, train_loss = engine.train_epoch(
                    self.train_loader, self.epoch, self.global_step
                )
                self._maybe_scheduler_step("step")  # for step-mode schedulers, step() has been called per step; safe no-op

                # Validation
                self.last_val_metrics = engine.validate(self.val_loader)
                metric_value = float(self.last_val_metrics.get(self.monitor, float("inf")))
                self.best_metric_value = min(self.best_metric_value, metric_value)

                # Save checkpoint
                self.ckpt.save(
                    self.model,
                    self.optimizer,
                    self.epoch,
                    self.global_step,
                    self.best_metric_value,
                    is_best=(metric_value <= self.best_metric_value),
                    scaler=self.scaler,
                    extra={"val_metrics": self.last_val_metrics, "cfg_hash": self.cfg_hash},
                )

                # EMA update after each epoch (if enabled)
                if self.ema is not None:
                    self.ema.update(self.model)

                # Callbacks
                self._callbacks("on_epoch_end", self._state())
                self._callbacks("on_validation_end", self._state())

                # Early stopping
                if self.early.step(metric_value):
                    logging.info(
                        "Early stopping at epoch %d with %s=%.6f",
                        self.epoch,
                        self.monitor,
                        metric_value,
                    )
                    break

        self._callbacks("on_train_end", self._state())
        barrier()
        return self.best_metric_value, self.last_val_metrics
