from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .loggers import JSONLLogger
from .utils import rank_zero_only


class Callback:
    """Base class for training callbacks."""

    def on_train_start(self, state: Dict[str, Any]) -> None: ...

    def on_epoch_start(self, state: Dict[str, Any]) -> None: ...

    def on_batch_end(self, state: Dict[str, Any]) -> None: ...

    def on_epoch_end(self, state: Dict[str, Any]) -> None: ...

    def on_validation_end(self, state: Dict[str, Any]) -> None: ...

    def on_train_end(self, state: Dict[str, Any]) -> None: ...


class JsonlCallback(Callback):
    """Write compact JSONL events after each batch and epoch."""

    def __init__(self, jsonl: JSONLLogger):
        self.jsonl = jsonl

    @rank_zero_only
    def on_train_start(self, state: Dict[str, Any]) -> None:
        self.jsonl.info(event="train_start", **state.get("run_meta", {}))

    @rank_zero_only
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        if "batch" in state and "loss" in state:
            self.jsonl.info(event="batch_end", step=state["global_step"], loss=float(state["loss"]))

    @rank_zero_only
    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        metrics = state.get("val_metrics", {})
        self.jsonl.info(event="epoch_end", epoch=state["epoch"], metrics=metrics)

    @rank_zero_only
    def on_train_end(self, state: Dict[str, Any]) -> None:
        self.jsonl.info(event="train_end", **state.get("final", {}))


class LRLoggerCallback(Callback):
    """Log learning rate at the end of each epoch."""

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        opt = state.get("optimizer")
        if opt and hasattr(opt, "param_groups"):
            lrs = [pg.get("lr", None) for pg in opt.param_groups]
            logging.info(
                "Epoch %d LR: %s",
                state.get("epoch", -1),
                ", ".join(f"{lr:.3e}" for lr in lrs if lr is not None),
            )


class TimeMeterCallback(Callback):
    """Track time per epoch."""

    def __init__(self):
        self.t0: Optional[float] = None

    def on_epoch_start(self, state: Dict[str, Any]) -> None:
        self.t0 = time.time()

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        if self.t0 is not None:
            dt = time.time() - self.t0
            logging.info("Epoch %d duration: %.2fs", state.get("epoch", -1), dt)


class MlflowCallback(Callback):
    """Optional MLflow logger. Fail-soft if mlflow not installed."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        try:
            import mlflow  # noqa: F401
        except Exception:
            self.enabled = False

    def _mlf(self):
        import mlflow
        return mlflow

    def on_train_start(self, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        mlf = self._mlf()
        run_meta = state.get("run_meta", {})
        mlf.start_run()
        mlf.set_tags(run_meta)
        logging.info("MLflow run started with tags: %s", run_meta)

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        mlf = self._mlf()
        metrics = state.get("val_metrics", {})
        for k, v in metrics.items():
            mlf.log_metric(k, float(v), step=int(state.get("epoch", 0)))

    def on_train_end(self, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        mlf = self._mlf()
        mlf.end_run()
        logging.info("MLflow run ended.")
