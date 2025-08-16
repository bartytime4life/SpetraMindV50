"""
Core Logging System
-------------------
Provides rotating file + console logging with Hydra-safe config.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import LoggingConfig
from .jsonl_handler import JSONLHandler

_loggers = {}


def init_logging(cfg: LoggingConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = Path(cfg.log_dir) / "spectramind.log"

    root = logging.getLogger()
    root.setLevel(cfg.log_level)
    root.handlers.clear()

    if cfg.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s"
        ))
        root.addHandler(console_handler)

    if cfg.file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=cfg.file_max_mb * 1024 * 1024,
            backupCount=cfg.file_backup_count
        )
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        ))
        root.addHandler(file_handler)

    if cfg.jsonl:
        jsonl_file = Path(cfg.log_dir) / "events.jsonl"
        root.addHandler(JSONLHandler(jsonl_file))


def get_logger(name: str) -> logging.Logger:
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]
