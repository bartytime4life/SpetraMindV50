from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

_JSONL_NAME = "train_events.jsonl"


def get_logger(name: str = "spectramind.train", level: int = logging.INFO) -> logging.Logger:
    """Create a console + rotating-file logger with safe defaults."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Rotating file handler
    Path("logs").mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        "logs/train.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(fh)

    logger.propagate = False
    logger.debug("Logger initialized with console + rotating file handlers.")
    return logger


def write_jsonl(event: Dict[str, Any], jsonl_path: str = _JSONL_NAME) -> None:
    """Append a structured event to a JSONL file for dashboards."""
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def try_mlflow_start(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
    """Start an MLflow run if mlflow is available; otherwise noop."""
    try:
        import mlflow  # type: ignore

        mlflow.set_experiment("SpectraMindV50")
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
    except Exception:
        pass


def try_mlflow_log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow if available; otherwise noop."""
    try:
        import mlflow  # type: ignore

        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def try_mlflow_log_params(params: Dict[str, Any]) -> None:
    """Log params to MLflow if available; otherwise noop."""
    try:
        import mlflow  # type: ignore

        mlflow.log_params(params)
    except Exception:
        pass


def try_mlflow_end() -> None:
    """End MLflow run if active; otherwise noop."""
    try:
        import mlflow  # type: ignore

        mlflow.end_run()
    except Exception:
        pass

