"""Logging helpers for the weights subsystem."""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

DEFAULT_LOG_DIR = Path(os.environ.get("SPECTRAMIND_LOG_DIR", "logs"))
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

TEXT_LOG = DEFAULT_LOG_DIR / "symbolic_weights.log"
EVENT_LOG = DEFAULT_LOG_DIR / "symbolic_weights.events.jsonl"

_logger_cache: Dict[str, logging.Logger] = {}


def get_logger(name: str = "spectramind.symbolic.weights") -> logging.Logger:
    if name in _logger_cache:
        return _logger_cache[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid double handlers if re-imported
    if not logger.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(ch)

        # Rotating file
        fh = RotatingFileHandler(
            TEXT_LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
        )
        logger.addHandler(fh)

    _logger_cache[name] = logger
    return logger


def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        rec = {"type": event_type, "payload": payload}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
