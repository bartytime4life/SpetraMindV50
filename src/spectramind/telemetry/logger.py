"""Minimal telemetry logger used for unit tests.

This module intentionally implements only a very small subset of the
telemetry features present in the full SpectraMind project.  The goal of this
lightweight implementation is to provide a stable, thread‑safe logging API that
the tests in this kata can exercise.  The :class:`TelemetryLogger` writes plain
text logs using Python's :mod:`logging` module and simultaneously stores
structured events in a JSONL file.  Both destinations are controlled via
environment variables so the tests can sandbox all generated artefacts.

Supported environment variables
-------------------------------

``SPECTRAMIND_TELEMETRY_LOG_DIR``
    Directory where log files will be written.  Falls back to
    ``SPECTRAMIND_LOG_DIR`` or the current directory if unset.

``SPECTRAMIND_TELEMETRY_JSONL_BASENAME``
    Name of the JSONL file (default ``telemetry.jsonl``).

``SPECTRAMIND_TELEMETRY_LOG_BASENAME``
    Name of the text log file (default ``telemetry.log``).

``SPECTRAMIND_TELEMETRY_ROTATE_MAX_MB`` and
``SPECTRAMIND_TELEMETRY_ROTATE_BACKUPS``
    Parameters controlling :class:`logging.handlers.RotatingFileHandler`.

The logger exposes a very small API surface: standard ``info``/``debug``/``error``
methods, an ``emit`` method for structured dictionaries and a ``bind`` method to
attach extra context that is included in every subsequent event.  All JSONL
writes are guarded by a :class:`threading.Lock` to guarantee thread‑safe
operation which is important for the concurrency tests.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def _log_dir() -> Path:
    """Resolve the directory where telemetry artefacts should be written."""

    base = os.environ.get("SPECTRAMIND_TELEMETRY_LOG_DIR") or os.environ.get(
        "SPECTRAMIND_LOG_DIR", "."
    )
    return Path(base)


class TelemetryLogger:
    """Small structured logger used throughout the tests.

    Parameters are intentionally very small; the logger relies almost entirely
    on environment variables to control behaviour so that the tests can sandbox
    outputs.  A new instance can be created by simply calling ``TelemetryLogger``
    or via :func:`get_telemetry_logger`.
    """

    def __init__(
        self,
        *,
        jsonl_path: Optional[str | Path] = None,
        log_path: Optional[str | Path] = None,
        context: Optional[Dict[str, Any]] = None,
        _logger: Optional[logging.Logger] = None,
        _lock: Optional[threading.Lock] = None,
    ) -> None:
        log_dir = _log_dir()
        jsonl_name = os.environ.get(
            "SPECTRAMIND_TELEMETRY_JSONL_BASENAME", "telemetry.jsonl"
        )
        log_name = os.environ.get(
            "SPECTRAMIND_TELEMETRY_LOG_BASENAME", "telemetry.log"
        )

        self.jsonl_path = Path(jsonl_path or (log_dir / jsonl_name))
        self.log_path = Path(log_path or (log_dir / log_name))

        # Ensure directories exist before handlers attempt to write.
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.run_id = os.environ.get("SPECTRAMIND_RUN_ID", "" )
        self.context: Dict[str, Any] = dict(context or {})
        self._lock = _lock or threading.Lock()

        if _logger is None:
            # Configure a dedicated logger instance with rotating file handling.
            self._logger = logging.Logger(f"telemetry-{id(self)}")
            self._logger.setLevel(logging.INFO)

            max_mb = float(os.environ.get("SPECTRAMIND_TELEMETRY_ROTATE_MAX_MB", "1"))
            max_bytes = int(max_mb * 1024 * 1024)
            backups = int(os.environ.get("SPECTRAMIND_TELEMETRY_ROTATE_BACKUPS", "3"))

            file_handler = RotatingFileHandler(
                self.log_path, maxBytes=max_bytes, backupCount=backups
            )
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(fmt)
            self._logger.addHandler(file_handler)

            if os.environ.get("SPECTRAMIND_TELEMETRY_ENABLE_CONSOLE", "1") == "1":
                console = logging.StreamHandler()
                console.setFormatter(fmt)
                self._logger.addHandler(console)

            self._logger.propagate = False
        else:
            # Reuse handlers from an existing logger when coming from bind().
            self._logger = _logger

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.time()

    def _prepare_event(
        self, level: str, message: str, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "timestamp": self._now(),
            "level": level,
            "message": message,
            "run_id": self.run_id,
        }
        event.update(self.context)
        if extra:
            event.update(extra)
        # Tests rely on a component field being present; provide a default.
        event.setdefault("component", self.context.get("component", "telemetry"))
        return event

    def _write_jsonl(self, event: Dict[str, Any]) -> None:
        """Append a single JSON event to the JSONL file."""

        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # public logging API
    # ------------------------------------------------------------------
    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """Generic logging entry point used by convenience wrappers."""

        extra = kwargs.get("extra")
        event = self._prepare_event(level.upper(), message, extra)
        self._write_jsonl(event)

        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:  # pragma: no cover - thin wrapper
        self.log("INFO", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:  # pragma: no cover
        self.log("DEBUG", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:  # pragma: no cover
        self.log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:  # pragma: no cover
        self.log("WARNING", message, **kwargs)

    def emit(self, payload: Dict[str, Any]) -> None:
        """Emit a pre‑structured event.

        The dictionary is merged on top of the base event structure so that the
        required fields are always present.
        """

        level = payload.get("level", "INFO")
        message = payload.get("message", "")
        event = self._prepare_event(level.upper(), message, payload)
        self._write_jsonl(event)

        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(message)

    # ------------------------------------------------------------------
    # context handling
    # ------------------------------------------------------------------
    def bind(self, **ctx: Any) -> "TelemetryLogger":
        """Return a new logger with additional context bound to all events."""

        new_ctx = dict(self.context)
        new_ctx.update(ctx)
        return TelemetryLogger(
            jsonl_path=self.jsonl_path,
            log_path=self.log_path,
            context=new_ctx,
            _logger=self._logger,
            _lock=self._lock,
        )

    def update_context(self, **ctx: Any) -> None:
        """Mutate the current logger's context in-place."""

        self.context.update(ctx)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close all handlers associated with the logger."""

        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


def get_telemetry_logger() -> TelemetryLogger:
    """Factory function used by the tests to obtain a new logger instance."""

    return TelemetryLogger()


# Some test discovery code also looks for a generic ``get_logger`` symbol.  We
# simply delegate to :func:`get_telemetry_logger` so either name works.
def get_logger(_: str | None = None) -> TelemetryLogger:  # pragma: no cover - tiny wrapper
    return get_telemetry_logger()


__all__ = ["TelemetryLogger", "get_telemetry_logger", "get_logger"]

