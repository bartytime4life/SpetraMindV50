import getpass
import json
import logging
import logging.handlers
import os
import pathlib
import socket
import sys
import time
import uuid
from typing import Any, Dict, Optional

DEFAULT_LOG_DIR = os.environ.get("SM_LOG_DIR", os.path.join(pathlib.Path.cwd(), "logs"))
DEFAULT_LOG_NAME = os.environ.get("SM_LOG_NAME", "spectramind_v50")
DEFAULT_VERBOSITY = int(os.environ.get("SM_VERBOSITY", "20"))  # INFO


class JsonlEventLogger:
    """Lightweight JSONL event stream writer with rotation."""

    def __init__(
        self, path: str, max_bytes: int = 20_000_000, backup_count: int = 10
    ) -> None:
        self.path = path
        self.handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        self._logger = logging.getLogger("spectramind.jsonl")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.addHandler(self.handler)

    def write(self, event: Dict[str, Any]) -> None:
        """Write a JSON event to the log."""
        try:
            event = dict(event) if event is not None else {}
            event.setdefault("ts", time.time())
            event.setdefault("host", socket.gethostname())
            event.setdefault("user", getpass.getuser())
            event.setdefault("pid", os.getpid())
            line = json.dumps(event, ensure_ascii=False)
            self._logger.info(line)
        except Exception as exc:  # pragma: no cover - best effort logging
            sys.stderr.write(f"[JsonlEventLogger] failed: {exc}\n")


def _ensure_dir(path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_logger(
    name: str = DEFAULT_LOG_NAME,
    log_dir: Optional[str] = None,
    level: int = DEFAULT_VERBOSITY,
    jsonl_name: Optional[str] = None,
):
    log_dir = log_dir or DEFAULT_LOG_DIR
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    jsonl_path = os.path.join(log_dir, f"{jsonl_name or (name + '.jsonl')}")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)

    jsonl = JsonlEventLogger(jsonl_path)
    return logger, jsonl, {"log_path": log_path, "jsonl_path": jsonl_path}


def log_event(jsonl: JsonlEventLogger, kind: str, data: Dict[str, Any]) -> None:
    """Log a structured event."""
    event = {"kind": kind, "event_id": str(uuid.uuid4()), **data}
    jsonl.write(event)
