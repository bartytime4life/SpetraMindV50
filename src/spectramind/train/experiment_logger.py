import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import RunContext, ensure_dir

try:
    import mlflow
except Exception:  # pragma: no cover - mlflow optional
    mlflow = None  # type: ignore


class ExperimentLogger:
    """
    Mission-grade experiment logger with:
    - Console + rotating file logs
    - JSONL event stream
    - Optional MLflow run (if mlflow installed and enabled)
    - Environment capture & run context snapshot
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str = "runs",
        jsonl_name: str = "events.jsonl",
        text_log_name: str = "train.log",
        mlflow_enable: bool = False,
        mlflow_experiment: Optional[str] = None,
        console_level: int = logging.INFO,
    ) -> None:
        ensure_dir(log_dir)
        self.run_name = run_name
        self.log_dir = str(Path(log_dir).resolve())
        self.jsonl_path = str(Path(self.log_dir) / jsonl_name)
        self.text_log_path = str(Path(self.log_dir) / text_log_name)
        self.console_level = console_level

        # python logging
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        # Remove existing handlers to avoid duplication on repeated init
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)

        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
        self.logger.addHandler(ch)

        fh = RotatingFileHandler(self.text_log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

        # JSONL stream
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")

        # Environment capture
        self.ctx = RunContext.capture()
        self.log_event("run_start", {"context": self.ctx.__dict__, "run_name": run_name})

        # Optional MLflow
        self.mlflow_run = None
        if mlflow_enable and mlflow is not None:
            if mlflow_experiment:
                mlflow.set_experiment(mlflow_experiment)
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            self.logger.info("MLflow run started.")

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        rec = {"event": event, **payload}
        self._jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def log_params(self, params: Dict[str, Any]) -> None:
        self.log_event("params", {"params": params})
        if self.mlflow_run is not None and mlflow is not None:
            flat = flatten_dict(params)
            mlflow.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, split: str = "train") -> None:
        payload = {"metrics": metrics, "step": step, "split": split}
        self.log_event("metrics", payload)
        if self.mlflow_run is not None and mlflow is not None:
            mlflow.log_metrics(prefix_keys(metrics, f"{split}/"), step=step)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def artifact(self, path: str, name: Optional[str] = None) -> None:
        """
        Register an artifact path; if MLflow enabled, log it there also.
        """
        self.log_event("artifact", {"path": path, "name": name})
        if self.mlflow_run is not None and mlflow is not None:
            mlflow.log_artifact(path, artifact_path=name)

    def close(self) -> None:
        self.log_event("run_end", {"run_name": self.run_name})
        try:
            self._jsonl.flush()
            self._jsonl.close()
        except Exception:
            pass
        if self.mlflow_run is not None and mlflow is not None:
            mlflow.end_run()
            self.logger.info("MLflow run closed.")


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out

def prefix_keys(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}
