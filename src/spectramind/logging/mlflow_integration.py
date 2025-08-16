"""
MLflow Integration
------------------
Optional MLflow logging for metrics and params.
"""
import mlflow
from .config import LoggingConfig


class MLflowLogger:
    def __init__(self, cfg: LoggingConfig):
        self.cfg = cfg
        if cfg.mlflow:
            mlflow.set_experiment(cfg.experiment)
            self.run = mlflow.start_run(run_name=cfg.run_name)

    def log_metric(self, key: str, value: float, step: int) -> None:
        if self.cfg.mlflow:
            mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: str) -> None:
        if self.cfg.mlflow:
            mlflow.log_param(key, value)
