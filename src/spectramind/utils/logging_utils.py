import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Iterable

from .paths import get_default_paths, ensure_dir
from .jsonl import jsonl_append

_DEFAULT_LOG_LEVEL = os.environ.get("SPECTRAMIND_LOG_LEVEL", "INFO").upper()
_MAX_BYTES = int(os.environ.get("SPECTRAMIND_LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10MB
_BACKUP_COUNT = int(os.environ.get("SPECTRAMIND_LOG_BACKUP_COUNT", 3))


class JsonFormatter(logging.Formatter):
    """Formatter that emits JSON lines with consistent keys."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _console_handler(level: str) -> logging.Handler:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    ch.setFormatter(fmt)
    return ch


def _rotating_file_handler(path: str, level: str) -> logging.Handler:
    ensure_dir(os.path.dirname(path))
    fh = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    return fh


def _jsonl_file_handler(path: str, level: str) -> logging.Handler:
    ensure_dir(os.path.dirname(path))
    handler = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    return handler


def get_logger(name: str = "spectramind", level: Optional[str] = None) -> logging.Logger:
    """Get or create a logger with SpectraMind defaults (console + rotating file).

    The file targets are derived from get_default_paths().log_file and .jsonl_file.

    Args:
        name: Logger name.
        level: Log level name (e.g., "INFO"). Defaults to env/SPECTRAMIND_LOG_LEVEL.

    Returns:
        Configured logging.Logger.
    """
    lvl = (level or _DEFAULT_LOG_LEVEL).upper()
    logger = logging.getLogger(name)
    if getattr(logger, "_spectramind_configured", False):
        return logger

    logger.setLevel(lvl)

    # Console
    logger.addHandler(_console_handler(lvl))

    # Rotating text file & JSONL file
    paths = get_default_paths()
    logger.addHandler(_rotating_file_handler(paths["log_file"], lvl))
    logger.addHandler(_jsonl_file_handler(paths["jsonl_log_file"], lvl))

    logger.propagate = False
    logger._spectramind_configured = True  # type: ignore[attr-defined]
    return logger


def add_file_handler(logger: logging.Logger, filepath: str, level: Optional[str] = None) -> None:
    """Add an additional rotating-file handler to an existing logger."""
    lvl = (level or _DEFAULT_LOG_LEVEL).upper()
    logger.addHandler(_rotating_file_handler(filepath, lvl))


def add_jsonl_handler(logger: logging.Logger, filepath: str, level: Optional[str] = None) -> None:
    """Add an additional JSONL rotating-file handler to an existing logger."""
    lvl = (level or _DEFAULT_LOG_LEVEL).upper()
    logger.addHandler(_jsonl_file_handler(filepath, lvl))


def configure_logging(
    name: str = "spectramind",
    level: Optional[str] = None,
    extra_files: Optional[Iterable[str]] = None,
    jsonl_to_event_stream: bool = True,
) -> logging.Logger:
    """Configure and return a logger with defaults, optionally adding more file sinks.
    Also connects the logger to the JSONL event stream file if requested.

    Args:
        name: Logger name.
        level: Log level.
        extra_files: Optional list of additional filepaths for text logs.
        jsonl_to_event_stream: If True, event writes go to the canonical JSONL log (EventStream).

    Returns:
        logger
    """
    logger = get_logger(name=name, level=level)
    if extra_files:
        for f in extra_files:
            add_file_handler(logger, f)

    if jsonl_to_event_stream:
        # Write a "logger-configured" event to the JSONL stream
        from .event_stream import EventStream

        paths = get_default_paths()
        es = EventStream(paths["event_log_file"])
        es.emit(event="logger-configured", payload={"logger": name, "level": (level or _DEFAULT_LOG_LEVEL).upper()})

    return logger
