from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .common import save_checkpoint
from .logger import get_logger, write_jsonl


@dataclass
class EarlyStopper:
    """Simple early stopping utility based on a monitored metric (lower is better)."""

    patience: int = 10
    min_delta: float = 0.0
    best: float = math.inf
    hits: int = 0
    stopped: bool = False
    metric_name: str = "val_loss"

    def step(self, value: float) -> bool:
        """Update with a new validation metric. Returns True if should stop."""
        improved = (self.best - value) > self.min_delta
        if improved:
            self.best = value
            self.hits = 0
        else:
            self.hits += 1
            if self.hits >= self.patience:
                self.stopped = True
        return self.stopped


class Checkpointer:
    """Manage periodic/best checkpoint saving and artifact manifest logging."""

    def __init__(self, out_dir: str, keep_last: int = 5, best_name: str = "best.pt"):
        self.out_dir = out_dir
        self.keep_last = keep_last
        self.best_name = best_name
        self.best_value = float("inf")
        self.logger = get_logger()

    def save_periodic(self, state: Dict[str, Any], tag: str) -> str:
        """Save a rolling checkpoint e.g., per-epoch."""
        fpath = save_checkpoint(state, self.out_dir, tag=tag, keep_last=self.keep_last)
        self.logger.info(f"Saved periodic checkpoint: {fpath}")
        write_jsonl({"event": "checkpoint_periodic", "path": fpath, "tag": tag})
        return fpath

    def try_save_best(self, state: Dict[str, Any], value: float) -> Optional[str]:
        """If `value` improves the best metric, save a 'best.pt' style checkpoint."""
        if value < self.best_value:
            self.best_value = value
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            fpath = str(Path(self.out_dir) / self.best_name)
            # Overwrite best
            import torch  # local import to keep module import light

            torch.save(state, fpath)
            self.logger.info(f"New best ({value:.6f}) saved to: {fpath}")
            write_jsonl({"event": "checkpoint_best", "path": fpath, "best": value})
            return fpath
        return None

