# SPDX-License-Identifier: MIT
"""JSONL logging handler for SpectraMind.

Writes one JSON object per log record to a ``.jsonl`` file. Each line contains
standard logging metadata plus any extra attributes attached to the record.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Standard keys present on ``logging.LogRecord`` objects. Attributes outside
# this set will be collected into an ``extras`` dictionary.
_JSONL_KEYS_STD = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


def _safe(obj: Any) -> Any:
    """Best-effort JSON serialisation."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _record_to_json(record: logging.LogRecord) -> Dict[str, Any]:
    ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "timestamp": ts,
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
        "path": record.pathname,
        "lineno": record.lineno,
        "func": record.funcName,
        "process": record.process,
        "thread": record.threadName,
        "run_id": getattr(record, "run_id", None),
        "git_hash": getattr(record, "git_hash", None),
    }
    extras: Dict[str, Any] = {}
    for k, v in record.__dict__.items():
        if k not in _JSONL_KEYS_STD and k not in payload and not k.startswith("_"):
            extras[k] = _safe(v)
    if extras:
        payload["extras"] = extras
    if record.exc_info:
        try:
            payload["exc_info"] = logging.Formatter().formatException(record.exc_info)
        except Exception:  # pragma: no cover - extremely rare
            payload["exc_info"] = "unavailable"
    return payload


class JSONLHandler(logging.Handler):
    """Logging handler writing records as JSON objects."""

    def __init__(self, filepath: Path, indent: int | None = None) -> None:
        super().__init__()
        self._path = Path(filepath)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        self._indent = indent

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - delegate
        try:
            obj = _record_to_json(record)
            text = json.dumps(obj, ensure_ascii=False, indent=self._indent)
            self._fh.write(text + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:  # pragma: no cover - simple delegation
        try:
            if not self._fh.closed:
                self._fh.close()
        finally:
            super().close()
