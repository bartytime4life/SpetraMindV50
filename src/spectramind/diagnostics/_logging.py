import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

_LOGGERS: Dict[str, logging.Logger] = {}

DEFAULT_LOG_DIR = os.environ.get("SPECTRAMIND_LOG_DIR", "logs")
DEFAULT_LOG_FILE = "v50_diagnostics.log"
DEFAULT_JSONL_FILE = "v50_diag_events.jsonl"


def _safe_makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def get_logger(
    name: str = "spectramind.diagnostics",
    log_dir: str = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or get a configured logger with console and rotating file handlers."""
    global _LOGGERS
    key = f"{name}|{log_dir}|{log_file}|{level}"
    if key in _LOGGERS:
        return _LOGGERS[key]

    _safe_makedirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture all; handlers filter levels

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        os.path.join(log_dir, log_file), maxBytes=10_000_000, backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(fh)

    logger.debug("Logger initialized: %s", key)
    _LOGGERS[key] = logger
    return logger


def _jsonl_path(log_dir: str) -> str:
    _safe_makedirs(log_dir)
    return os.path.join(log_dir, DEFAULT_JSONL_FILE)


def log_event(
    event: str,
    details: Optional[Dict[str, Any]] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Append a JSONL event to the diagnostics event stream."""
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "event": event,
        "details": details or {},
    }
    try:
        with open(_jsonl_path(log_dir), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning("Failed to write JSONL event: %s", e)
    if logger:
        logger.info("[EVENT] %s :: %s", event, details or {})


def capture_env_and_git(
    log_dir: str = DEFAULT_LOG_DIR,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Capture minimal ENV and Git snapshot for reproducibility."""
    snap: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "env_subset": {
            k: os.environ.get(k, "")
            for k in [
                "CONDA_DEFAULT_ENV",
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "SPECTRAMIND_LOG_DIR",
                "MLFLOW_TRACKING_URI",
                "WANDB_PROJECT",
                "WANDB_MODE",
            ]
        },
        "git": {"commit": None, "branch": None, "dirty": None},
    }
    try:
        import subprocess

        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        dirty = bool(status.strip())
        snap["git"] = {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:  # noqa: BLE001
        pass

    try:
        with open(
            os.path.join(log_dir, "repro_snapshot.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(snap, f, indent=2)
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning("Failed to write repro snapshot: %s", e)

    if logger:
        logger.info("Repro snapshot: %s", snap["git"])
    log_event("repro-snapshot", snap, log_dir=log_dir, logger=logger)
    return snap


def maybe_mlflow_start_run(
    enabled: bool,
    run_name: str,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Optional MLflow integration. Returns a context manager or no-op."""
    if not enabled:

        class _Noop:
            def __enter__(self) -> None:  # noqa: D401
                return None

            def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401, ANN001
                return False

        if logger:
            logger.debug("MLflow disabled; using no-op.")
        return _Noop()
    try:
        import mlflow

        if logger:
            logger.info("Starting MLflow run: %s", run_name)
        return mlflow.start_run(run_name=run_name)
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning("MLflow unavailable or failed to start run: %s", e)

        class _Noop:
            def __enter__(self) -> None:  # noqa: D401
                return None

            def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401, ANN001
                return False

        return _Noop()


def maybe_wandb_init(
    enabled: bool,
    project: str,
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Optional Weights & Biases integration. Returns a run object or no-op."""
    if not enabled:
        if logger:
            logger.debug("Weights & Biases disabled; using no-op.")

        class _NoWandb:
            def log(self, *a, **k) -> None:  # noqa: ANN001, D401
                pass

            def finish(self) -> None:  # noqa: D401
                pass

        return _NoWandb()
    try:
        import wandb

        mode = os.environ.get("WANDB_MODE", "online")
        if logger:
            logger.info(
                "Initializing W&B: project=%s, run=%s, mode=%s", project, run_name, mode
            )
        return wandb.init(
            project=project, name=run_name, config=config or {}, mode=mode
        )
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning("W&B unavailable or failed to init: %s", e)

        class _NoWandb:
            def log(self, *a, **k) -> None:  # noqa: ANN001, D401
                pass

            def finish(self) -> None:  # noqa: D401
                pass

        return _NoWandb()
