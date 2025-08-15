"""Centralised logging utilities for the calibration package.

Provides console + rotating file handlers and a JSONL event stream hook.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_LOG_DIR = Path(os.getenv("SPECTRAMIND_LOG_DIR", "logs"))
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "calibration.log"
DEFAULT_JSONL_FILE = DEFAULT_LOG_DIR / "calibration_events.jsonl"
DEFAULT_DEBUG_MD = DEFAULT_LOG_DIR / "v50_debug_log.md"


@dataclass
class LogConfig:
    """Configuration for :func:`setup_logging`."""

    name: str = "spectramind.calibration"
    level: int = logging.INFO
    file_path: Path = DEFAULT_LOG_FILE
    max_bytes: int = 5_000_000
    backup_count: int = 5
    jsonl_path: Path = DEFAULT_JSONL_FILE
    debug_md_path: Path = DEFAULT_DEBUG_MD


class JSONLEventLogger:
    """Minimal JSONL logger used for diagnostics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, event: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover - best effort
            pass


def setup_logging(cfg: Optional[LogConfig] = None) -> Tuple[logging.Logger, JSONLEventLogger]:
    """Create a logger and JSONL event logger."""

    cfg = cfg or LogConfig()
    logger = logging.getLogger(cfg.name)
    logger.setLevel(cfg.level)
    logger.propagate = False

    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(cfg.level)
        console.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(console)

        # Rotating file handler
        file_handler = RotatingFileHandler(
            filename=str(cfg.file_path),
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(cfg.level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    # JSONL event stream
    evt = JSONLEventLogger(cfg.jsonl_path)
    return logger, evt


def append_debug_md(header: str, details: Dict[str, Any], path: Path = DEFAULT_DEBUG_MD) -> None:
    """Append a diagnostic block to ``v50_debug_log.md``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {header}\n\n")
        for k, v in details.items():
            f.write(f"- {k}: {v}\n")
