from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from .losses import compute_total_loss
from .metrics import compute_metrics
from .utils import to_device


class TrainEngine:
    """
    Handles per-epoch train and validation loops with AMP, grad accumulation, and metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: "torch.cuda.amp.GradScaler | None",
        cfg: Dict[str, Any],
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.cfg = cfg

        self.grad_accum = int(cfg.get("training", {}).get("grad_accum", 1))
        self.max_grad_norm = float(cfg.get("training", {}).get("max_grad_norm", 0.0))
        self.use_amp = bool(cfg.get("training", {}).get("amp", True))

    def _forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Expects model forward signature compatible with batch.
        Batch should include keys needed by model; typical:
          batch["inputs"] -> model(inputs) returns dict with "mu", "sigma"
        """
        outputs = self.model(batch)  # model is expected to handle dict batch
        if not isinstance(outputs, dict) or "mu" not in outputs:
            raise RuntimeError("Model forward must return dict with at least key 'mu'")
        return outputs

    def train_epoch(self, loader: Iterable, epoch: int, global_step: int) -> Tuple[int, float]:
        self.model.train()
        running = 0.0
        steps = 0
        optimizer = self.optimizer

        for i, batch in enumerate(loader):
            batch = to_device(batch, self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                outputs = self._forward(batch)
                target = batch.get("target") or batch.get("y") or batch.get("labels")
                if target is None:
                    raise RuntimeError("Batch must contain 'target' or 'y' or 'labels'")
                meta = batch.get("meta", None)
                losses = compute_total_loss(outputs, target, meta, self.cfg.get("loss", {}))
                loss = losses["total"] / self.grad_accum

            if self.scaler is not None and self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.grad_accum == 0:
                if self.scaler is not None and self.use_amp:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(optimizer)
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += float(losses["total"].detach().item())
            steps += 1

        avg_loss = running / max(steps, 1)
        logging.info("Train epoch %d: loss=%.6f (%d steps)", epoch, avg_loss, steps)
        return global_step, avg_loss

    @torch.no_grad()
    def validate(self, loader: Iterable) -> Dict[str, float]:
        self.model.eval()
        agg = {"gll": 0.0, "rmse": 0.0, "mae": 0.0, "coverage68": 0.0}
        n = 0
        for batch in loader:
            batch = to_device(batch, self.device)
            outputs = self._forward(batch)
            target = batch.get("target") or batch.get("y") or batch.get("labels")
            metrics = compute_metrics(outputs, target)
            for k, v in metrics.items():
                agg[k] += float(v)
            n += 1
        return {k: (v / max(n, 1)) for k, v in agg.items()}
