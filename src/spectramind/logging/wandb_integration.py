"""
Weights & Biases Integration
----------------------------
Optional WandB logging for metrics and telemetry.
"""
import wandb
from .config import LoggingConfig


class WandBLogger:
    def __init__(self, cfg: LoggingConfig):
        self.cfg = cfg
        if cfg.wandb:
            wandb.init(project=cfg.project, name=cfg.run_name)

    def log(self, data: dict, step: int = None) -> None:
        if self.cfg.wandb:
            wandb.log(data, step=step)
