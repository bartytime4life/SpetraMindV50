# SPDX-License-Identifier: MIT

"""Logging utilities for configuration helpers."""

import os
import sys
import json
import time
import atexit
import socket
import platform
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

V50_DEBUG_LOG = Path("v50_debug_log.md")
LOG_JSONL = LOGS_DIR / "event_stream.jsonl"
ROTATING_LOG = LOGS_DIR / "spectramind.log"

ENABLE_MLFLOW = os.getenv("SMV50_MLFLOW", "0") in ("1", "true", "True", "YES", "yes")
ENABLE_WANDB = os.getenv("SMV50_WANDB", "0") in ("1", "true", "True", "YES", "yes")

_LOGGER: Optional[logging.Logger] = None


def _ensure_md_header() -> None:
    if not V50_DEBUG_LOG.exists():
        with open(V50_DEBUG_LOG, "w", encoding="utf-8") as f:
            f.write("# SpectraMind V50 Debug Log\n\n")


def init_logging(level: int = logging.INFO) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    _ensure_md_header()

    logger = logging.getLogger("spectramind.conf_helpers")
    logger.setLevel(level)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    fh = RotatingFileHandler(ROTATING_LOG, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"))
    logger.addHandler(fh)

    atexit.register(lambda: [h.flush() for h in logger.handlers if hasattr(h, "flush")])

    _LOGGER = logger

    host = socket.gethostname()
    sysinfo = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "host": host,
    }
    write_md("Logger initialized", sysinfo)
    log_event("logger_initialized", sysinfo)

    if ENABLE_MLFLOW:
        try:  # pragma: no cover - optional dependency
            import mlflow  # type: ignore

            mlflow.set_experiment("SpectraMindV50")
            log_event("mlflow_enabled", {})
        except Exception as e:  # pragma: no cover
            logger.warning("MLflow init failed: %s", e)
    if ENABLE_WANDB:
        try:  # pragma: no cover - optional dependency
            import wandb  # type: ignore

            wandb.init(
                project="SpectraMindV50",
                reinit=True,
                mode="offline" if os.getenv("WANDB_MODE") == "offline" else "online",
            )
            log_event("wandb_enabled", {})
        except Exception as e:  # pragma: no cover
            logger.warning("W&B init failed: %s", e)

    return logger


def get_logger() -> logging.Logger:
    return init_logging()


def write_md(action: str, meta: Optional[Dict[str, Any]] = None) -> None:
    V50_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = f"### {ts} — {action}\n"
    if meta is not None:
        entry += "```json\n" + json.dumps(meta, indent=2, sort_keys=True) + "\n```\n"
    with open(V50_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(entry)


def log_event(event: str, payload: Dict[str, Any]) -> None:
    LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {"ts": ts, "event": event, "payload": payload}
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    logger = get_logger()
    logger.info("event=%s payload=%s", event, payload)
