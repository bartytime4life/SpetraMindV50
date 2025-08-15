from __future__ import annotations
import os
from typing import Optional, Dict, Any


class MLflowTracker:
    def __init__(self, enabled: bool, tracking_uri: Optional[str], experiment: str):
        self.enabled = enabled
        if not enabled:
            return
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run()
        self.mlflow = mlflow

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.enabled:
            self.mlflow.log_metrics(metrics, step=step)

    def end(self):
        if self.enabled:
            self.mlflow.end_run()


class WandbTracker:
    def __init__(self, enabled: bool, project: str, entity: Optional[str] = None):
        self.enabled = enabled
        if not enabled:
            return
        import wandb
        wandb.init(project=project, entity=entity, settings=wandb.Settings(start_method="thread"))
        self.wandb = wandb

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.enabled:
            self.wandb.log(metrics | ({"step": step} if step is not None else {}))

    def end(self):
        if self.enabled:
            self.wandb.finish()

