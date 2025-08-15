import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(os.environ.get("SPECTRAMIND_LOG_DIR", "logs")).resolve()
LOG_DIR.mkdir(exist_ok=True, parents=True)

DEBUG_LOG_FILE = LOG_DIR / "v50_debug_log.md"
EVENTS_JSONL_FILE = LOG_DIR / "events.jsonl"
ROTATING_LOG_FILE = LOG_DIR / "spectramind.log"

ROTATE_BYTES = int(
    os.environ.get("SPECTRAMIND_LOG_ROTATE_BYTES", str(10 * 1024 * 1024))
)
ROTATE_BACKUPS = int(os.environ.get("SPECTRAMIND_LOG_ROTATE_BACKUPS", "5"))


def setup_rotating_handlers(logger: logging.Logger, level: int) -> None:
    """
    Attach console + rotating file handlers with consistent formatting.
    Safe to call multiple times; handlers are only added once per logger.
    """
    # Check if already configured
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers) and any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        return

    logger.setLevel(level)

    # Console handler (UTC ISO timestamps in stream for consistent CI parsing)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        logging.Formatter(
            "[%(asctime)sZ] %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
    )
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        ROTATING_LOG_FILE,
        maxBytes=ROTATE_BYTES,
        backupCount=ROTATE_BACKUPS,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)sZ [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(fh)


def get_logger(
    name: str = "spectramind", level: Optional[int] = None
) -> logging.Logger:
    """
    Create or retrieve a configured logger with console + rotating file output.
    Level defaults to INFO; override via SPECTRAMIND_LOG_LEVEL or parameter.
    """
    logger = logging.getLogger(name)
    # Determine desired level: parameter > ENV > INFO
    level_final = (
        level
        if level is not None
        else getattr(
            logging,
            os.environ.get("SPECTRAMIND_LOG_LEVEL", "INFO").upper(),
            logging.INFO,
        )
    )
    setup_rotating_handlers(logger, level_final)
    return logger


def _append_markdown_row(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Append a structured row to the Markdown debug log.
    """
    ts = datetime.utcnow().isoformat()
    # Create header if missing
    if not DEBUG_LOG_FILE.exists() or DEBUG_LOG_FILE.stat().st_size == 0:
        with open(DEBUG_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("| timestamp_utc | event | payload_json |\n")
            f.write("|---|---|---|\n")
    # Append the row
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"| {ts} | {event_type} | {json.dumps(payload, sort_keys=True)} |\n")


def _append_jsonl(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Append a JSONL event record for programmatic ingestion and dashboards.
    """
    record = {"ts": datetime.utcnow().isoformat(), "type": event_type, "data": payload}
    with open(EVENTS_JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def log_event(
    event_type: str,
    payload: Dict[str, Any],
    *,
    also_jsonl: bool = True,
    also_md: bool = True,
) -> None:
    """
    Record a structured event to both Markdown and JSONL streams.
    """
    if also_md:
        _append_markdown_row(event_type, payload)
    if also_jsonl:
        _append_jsonl(event_type, payload)


def log_cli_call(
    command: str,
    args: Dict[str, Any],
    config_hash: str,
    cli_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Specialized recorder for CLI invocations. Consumed by analyze-log / dashboard.
    """
    payload = {
        "command": command,
        "args": args,
        "config_hash": config_hash,
        "cli_version": cli_version,
    }
    if extra:
        payload.update(extra)
    log_event("cli_call", payload)
