# SPDX-License-Identifier: MIT
"""Core logging utilities for SpectraMind."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LoggingConfig
from .jsonl_handler import JSONLHandler

LOG_DIR = Path("logs")
DEBUG_LOG_FILE = LOG_DIR / "v50_debug_log.md"

# Cache of named loggers for quick reuse
_LOGGERS: Dict[str, logging.Logger] = {}


class _ContextFilter(logging.Filter):
    """Inject reproducibility context into every log record."""

    def __init__(self, run_id: str, git_hash: str) -> None:
        super().__init__()
        self._run_id = run_id
        self._git_hash = git_hash

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, "run_id", self._run_id)
        setattr(record, "git_hash", self._git_hash)
        return True


def get_git_hash() -> str:
    """Return short git hash of current repository, if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return out or "unknown"
    except Exception:  # pragma: no cover - git may be absent
        return "unknown"


def get_run_id() -> str:
    """Return a deterministic process identifier."""
    return f"pid{os.getpid()}@{Path.cwd().name}"


def add_context(
    logger: logging.Logger, run_id: Optional[str] = None, git_hash: Optional[str] = None
) -> None:
    """Attach the context filter to ``logger``."""
    run_id = run_id or get_run_id()
    git_hash = git_hash or get_git_hash()
    logger.addFilter(_ContextFilter(run_id=run_id, git_hash=git_hash))


def init_logging(cfg: LoggingConfig, extra_context: Optional[Dict[str, Any]] = None) -> None:
    """Initialise global logging according to ``cfg``."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "spectramind.log"
    jsonl_file = log_dir / "events.jsonl"

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    # Remove any pre-existing handlers to keep init idempotent
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # Shared reproducibility context
    run_id = get_run_id()
    git_hash = get_git_hash()
    ctx_filter = _ContextFilter(run_id=run_id, git_hash=git_hash)
    root.addFilter(ctx_filter)

    if cfg.console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(root.level)
        ch.setFormatter(logging.Formatter(cfg.console_fmt, datefmt=cfg.date_fmt))
        ch.addFilter(ctx_filter)
        root.addHandler(ch)

    if cfg.file:
        fh = RotatingFileHandler(
            filename=log_file,
            maxBytes=cfg.file_max_mb * 1024 * 1024,
            backupCount=cfg.file_backup_count,
            encoding="utf-8",
            delay=False,
        )
        fh.setLevel(root.level)
        fh.setFormatter(logging.Formatter(cfg.file_fmt, datefmt=cfg.date_fmt))
        fh.addFilter(ctx_filter)
        root.addHandler(fh)

    if cfg.jsonl:
        jh = JSONLHandler(filepath=jsonl_file, indent=cfg.jsonl_indent)
        jh.setLevel(root.level)
        jh.addFilter(ctx_filter)
        root.addHandler(jh)

    if extra_context:
        init_logger = logging.getLogger("spectramind.init")
        init_logger.info(
            "init_logging_context",
            extra={k: v for k, v in extra_context.items()},
        )


def get_logger(name: str) -> logging.Logger:
    """Return a cached logger with ``name``."""
    if name not in _LOGGERS:
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]


def log_event(event: str, payload: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """Log a structured event to the given logger (root by default)."""
    lg = logger or get_logger("spectramind")
    lg.info(event, extra=payload)


def log_cli_call(
    command: str,
    args: Dict[str, Any],
    config_hash: str,
    version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a structured CLI invocation event."""
    payload: Dict[str, Any] = {
        "command": command,
        "args": args,
        "config_hash": config_hash,
        "version": version,
    }
    if extra:
        payload.update(extra)
    get_logger("spectramind.cli").info("cli_call", extra={"cli": payload})
