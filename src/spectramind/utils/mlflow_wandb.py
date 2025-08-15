from __future__ import annotations

import os
from typing import Any, Dict, Optional


def try_mlflow_start(run_name: str, tags: Optional[Dict[str, Any]] = None):
    """Start an MLflow run if ``MLFLOW_TRACKING_URI`` is configured."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    try:  # pragma: no cover - optional dependency
        import mlflow

        mlflow.set_tracking_uri(uri)
        run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        return run
    except Exception:
        return None


def try_mlflow_log_metrics(
    metrics: Dict[str, float], step: Optional[int] = None
) -> None:
    try:  # pragma: no cover - optional dependency
        import mlflow

        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def try_mlflow_log_params(params: Dict[str, Any]) -> None:
    try:  # pragma: no cover - optional dependency
        import mlflow

        flat = {
            k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for k, v in params.items()
        }
        mlflow.log_params(flat)
    except Exception:
        pass


def try_wandb_init(
    run_name: str,
    project: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """Initialise a Weights & Biases run if ``WANDB_API_KEY`` is set."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:  # pragma: no cover - optional dependency
        import wandb

        return wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "spectramind-v50"),
            name=run_name,
            config=config or {},
        )
    except Exception:
        return None
