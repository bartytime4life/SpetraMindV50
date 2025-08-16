from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping, CheckpointManager
from .experiment_logger import ExperimentLogger


@dataclass
class TrainerHooks:
    """
    Optional hooks to customize per-epoch behavior.
    """
    on_epoch_start: Optional[Callable[[int], None]] = None
    on_epoch_end: Optional[Callable[[int, Dict[str, float], Dict[str, float]], None]] = None
    on_validation_end: Optional[Callable[[int, Dict[str, float]], None]] = None


class TrainerBase:
    """
    Mission-grade trainer with AMP, grad accumulation, early stopping, and checkpointing.
    Accepts a 'step_fn(model, batch, device) -> (loss_tensor, metrics_dict)' callable.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step_fn: Callable[[torch.nn.Module, Dict, torch.device], Tuple[torch.Tensor, Dict[str, float]]],
        device: torch.device,
        logger: ExperimentLogger,
        scheduler=None,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        clip_grad_norm: Optional[float] = None,
        checkpoint: Optional[CheckpointManager] = None,
        early_stopping: Optional[EarlyStopping] = None,
        hooks: Optional[TrainerHooks] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.device = device
        self.logger = logger
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.use_amp = use_amp
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.hooks = hooks or TrainerHooks()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _run_split(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        metrics_accum: Dict[str, float] = {}
        num_steps = 0

        for step, batch in enumerate(loader):
            if train:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, scalars = self.step_fn(self.model, batch, self.device)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()

                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None and hasattr(self.scheduler, "step") and not hasattr(self.scheduler, "step_on_epoch"):
                        self.scheduler.step()
            else:
                with torch.no_grad():
                    loss, scalars = self.step_fn(self.model, batch, self.device)

            total_loss += float(loss.detach().item())
            for k, v in scalars.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
            num_steps += 1

        # Normalize
        if num_steps > 0:
            total_loss /= num_steps
            for k in list(metrics_accum.keys()):
                metrics_accum[k] /= num_steps

        metrics_accum["loss"] = total_loss
        return metrics_accum

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        start_epoch: int = 1,
    ) -> Dict[str, float]:
        best_val = float("inf")
        for epoch in range(start_epoch, start_epoch + epochs):
            if self.hooks.on_epoch_start:
                self.hooks.on_epoch_start(epoch)

            train_metrics = self._run_split(train_loader, train=True)
            self.logger.log_metrics(train_metrics, step=epoch, split="train")

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._run_split(val_loader, train=False)
                self.logger.log_metrics(val_metrics, step=epoch, split="val")

            # epoch-level scheduler (if provided)
            if self.scheduler is not None and hasattr(self.scheduler, "step_on_epoch"):
                self.scheduler.step_on_epoch()  # custom semantics
            elif self.scheduler is not None and hasattr(self.scheduler, "step") and hasattr(self.scheduler, "step_on_epoch_only"):
                self.scheduler.step()

            if self.hooks.on_validation_end:
                self.hooks.on_validation_end(epoch, val_metrics)

            # checkpointing
            to_save = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            if self.checkpoint is not None:
                self.checkpoint.save_last(to_save)
                # monitor val loss by default
                monitor_val = val_metrics.get("loss", float("inf")) if val_metrics else train_metrics.get("loss", float("inf"))
                if monitor_val < best_val:
                    best_val = monitor_val
                    self.checkpoint.save_best(to_save)

            # early stopping
            if self.early_stopping is not None and val_metrics:
                should_stop = self.early_stopping.step({"val_loss": val_metrics.get("loss", float("inf"))})
                if should_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}. Best={self.early_stopping.best:.6f}")
                    break

            if self.hooks.on_epoch_end:
                self.hooks.on_epoch_end(epoch, train_metrics, val_metrics)

        return {"best_val_loss": best_val}
