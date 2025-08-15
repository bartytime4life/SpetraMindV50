import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlHandler(logging.Handler):
    """
    Minimal JSONL handler: each record -> one JSON object line.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            for k, v in record.__dict__.items():
                if k not in (
                    "args",
                    "created",
                    "exc_info",
                    "exc_text",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "msg",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "thread",
                    "threadName",
                ):
                    try:
                        json.dumps({k: v})
                        entry[k] = v
                    except Exception:
                        entry[k] = str(v)
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def setup_logging(
    run_name: str = "calibration",
    log_dir: str = "logs",
    log_file_basename: str = "v50_debug_log.log",
    jsonl_file_basename: str = "events.jsonl",
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> Dict[str, Any]:
    """
    Configure logging with:
    - console (INFO)
    - rotating file log
    - JSONL event stream
    Returns a dict of resolved paths and run metadata.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logfile_path = Path(log_dir) / log_file_basename
    jsonl_path = Path(log_dir) / jsonl_file_basename

    root = logging.getLogger()
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(ch)

        fh = RotatingFileHandler(
            str(logfile_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(fh)

        jh = JsonlHandler(jsonl_path)
        jh.setLevel(level)
        root.addHandler(jh)

    meta = {
        "run_name": run_name,
        "timestamp_utc": ts,
        "logfile": str(logfile_path),
        "jsonl": str(jsonl_path),
        "cwd": os.getcwd(),
    }
    logging.getLogger(__name__).info("Logging configured", extra={"meta": meta})
    return meta


def log_event(
    event_type: str, payload: Optional[Dict[str, Any]] = None, level: int = logging.INFO
) -> None:
    """
    Convenience: emit a structured event to JSONL via standard logging.
    """
    payload = payload or {}
    logging.getLogger("spectramind.events").log(
        level, f"event:{event_type}", extra=payload
    )
