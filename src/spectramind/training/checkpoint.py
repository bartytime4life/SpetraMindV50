from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .utils import ensure_dir, write_json


class CheckpointManager:
    """
    Save/Load checkpoints with metadata. Keeps "last.pt" and "best.pt".
    """

    def __init__(self, dirpath: Path | str):
        self.dir = ensure_dir(dirpath)
        self.last = Path(self.dir) / "last.pt"
        self.best = Path(self.dir) / "best.pt"
        self.meta_path = Path(self.dir) / "meta.json"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        best_metric_value: float,
        is_best: bool,
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_metric_value": best_metric_value,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "extra": extra or {},
        }
        torch.save(state, self.last)
        if is_best:
            torch.save(state, self.best)
        write_json(
            self.meta_path,
            {
                "epoch": epoch,
                "step": step,
                "best_metric_value": best_metric_value,
                "is_best": is_best,
                "last_path": str(self.last),
                "best_path": str(self.best) if is_best else None,
            },
        )
        logging.info("Checkpoint saved at epoch=%d step=%d (best=%s)", epoch, step, is_best)

    def load(self, path: Optional[Path | str] = None) -> Dict[str, Any]:
        ckpt_path = Path(path) if path is not None else self.last
        logging.info("Loading checkpoint from %s", ckpt_path)
        return torch.load(ckpt_path, map_location="cpu")
