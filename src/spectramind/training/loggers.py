from __future__ import annotations

import json
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir, rank_zero_only


def setup_logging(log_dir: Path | str, name: str = "spectramind") -> Path:
    """
    Configure console logger + rotating file logger.
    Returns the path to the primary log file.
    """
    log_dir = ensure_dir(log_dir)
    log_path = Path(log_dir) / f"{name}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplication in notebooks/CLI calls
    for h in list(logger.handlers):
        logger.removeHandler(h)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))

    rotate = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    rotate.setLevel(logging.INFO)
    rotate.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(console)
    logger.addHandler(rotate)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    return log_path


class JSONLLogger:
    """
    Lightweight JSONL event stream writer. Safe for rank-zero only.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    @rank_zero_only
    def write(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @rank_zero_only
    def info(self, **kwargs: Any) -> None:
        kwargs.setdefault("level", "INFO")
        self.write(kwargs)
