# SPDX-License-Identifier: MIT
"""Optional Weights & Biases integration for SpectraMind logging."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore
    _WANDB_OK = True
except Exception:  # pragma: no cover - W&B not installed
    wandb = None  # type: ignore
    _WANDB_OK = False

from .config import LoggingConfig


def wandb_available() -> bool:
    """Return True if W&B is importable."""
    return _WANDB_OK


class WandBLogger:
    """Thin wrapper around ``wandb`` logging functions."""

    def __init__(self, cfg: LoggingConfig):
        self.cfg = cfg
        self._active = bool(cfg.wandb and _WANDB_OK)
        if self._active:
            wandb.init(project=cfg.project, name=cfg.run_name)  # type: ignore[attr-defined]

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._active:
            wandb.log(data, step=step)  # type: ignore[attr-defined]

    def set_tags(self, **tags: Any) -> None:
        if self._active:
            run = wandb.run  # type: ignore[attr-defined]
            current = list(run.tags or [])
            run.tags = list(set(current + list(tags.values())))

    def finish(self) -> None:
        if self._active:
            wandb.finish()  # type: ignore[attr-defined]
            self._active = False
