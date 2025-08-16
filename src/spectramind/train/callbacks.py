from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional


@dataclass
class EarlyStopping:
    """
    Simple early-stopping on a monitored metric.
    """
    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    patience: int = 10
    best: Optional[float] = None
    num_bad: int = 0
    stopped: bool = False

    def step(self, metrics: Dict[str, float]) -> bool:
        """
        Update with latest metrics; return True if should stop.
        """
        current = metrics.get(self.monitor)
        if current is None:
            return False

        if self.best is None:
            self.best = current
            self.num_bad = 0
            return False

        improved = (current < self.best) if self.mode == "min" else (current > self.best)
        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.stopped = True
        return self.stopped


class CheckpointManager:
    """
    Manage checkpoint saving with 'best' and 'last' semantics.
    """
    def __init__(self, out_dir: str, best_name: str = "best.ckpt", last_name: str = "last.ckpt") -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.best_path = os.path.join(out_dir, best_name)
        self.last_path = os.path.join(out_dir, last_name)

    def save_last(self, state: Dict) -> str:
        import torch
        torch.save(state, self.last_path)
        return self.last_path

    def save_best(self, state: Dict) -> str:
        import torch
        torch.save(state, self.best_path)
        return self.best_path
