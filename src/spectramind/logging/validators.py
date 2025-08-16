# SPDX-License-Identifier: MIT
"""Logging validators for SpectraMind."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import LoggingConfig
from .logger import get_logger, init_logging


def _read_first_jsonl_line(jsonl_path: Path) -> Dict[str, Any]:
    if not jsonl_path.exists():
        raise AssertionError(f"JSONL file not found: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise AssertionError("JSONL file exists but first line is empty")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSONL: {e}") from e
    for k in ("timestamp", "level", "logger", "message"):
        if k not in obj:
            raise AssertionError(f"JSONL schema missing key: {k}")
    try:
        datetime.fromisoformat(obj["timestamp"].replace("Z", "+00:00"))
    except Exception as e:  # pragma: no cover - extremely rare
        raise AssertionError(f"Invalid timestamp format in JSONL: {e}") from e
    return obj


def _assert_file_rotating(
    log_dir: Path, base_name: str = "spectramind.log", max_bytes: int = 64 * 1024
) -> Tuple[Path, Optional[Path]]:
    primary = log_dir / base_name
    if not primary.exists():
        raise AssertionError(f"Primary log file not found: {primary}")
    lg = get_logger("spectramind.validator.rotation")
    msg = "ROTATION_TEST_MESSAGE" + ("x" * 1024)
    target_bytes = max_bytes + 20 * len(msg)
    written = 0
    i = 0
    while written < target_bytes and i < 5000:
        lg.info("%s_%06d", msg, i)
        written += len(msg) + 16
        i += 1
    time.sleep(0.25)
    rotated = None
    for suf in (".1", ".0", ".01"):
        c = log_dir / f"{base_name}{suf}"
        if c.exists():
            rotated = c
            break
    if rotated is None:
        lg.info("FORCE_ROTATE_TICK")
        time.sleep(0.1)
        for suf in (".1", ".0", ".01"):
            c = log_dir / f"{base_name}{suf}"
            if c.exists():
                rotated = c
                break
    if rotated is None:
        raise AssertionError("Expected rotated log file was not found after stress logging")
    return primary, rotated


def validate_logging_integrity(
    tmp_dir: Optional[str] = None,
    level: str = "DEBUG",
    check_rotation: bool = True,
    expected_console: bool = True,
    expected_file: bool = True,
    expected_jsonl: bool = True,
) -> Dict[str, Any]:
    """Run a battery of checks validating logging configuration."""
    out: Dict[str, Any] = {
        "name": "validate_logging_integrity",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tmp_dir": None,
        "checks": [],
        "status": "unknown",
    }
    log_dir = Path(tmp_dir) if tmp_dir is not None else Path(".") / "selftest_artifacts" / "logging"
    log_dir.mkdir(parents=True, exist_ok=True)
    out["tmp_dir"] = str(log_dir)

    cfg = LoggingConfig(
        log_level=level,
        log_dir=str(log_dir),
        console=expected_console,
        file=expected_file,
        jsonl=expected_jsonl,
        file_max_mb=1,
        file_backup_count=2,
        mlflow=False,
        wandb=False,
        project="spectramind-v50",
        run_name="selftest-logging",
        experiment="selftest",
    )
    init_logging(cfg)
    # Speed up rotation during validation
    for h in logging.getLogger().handlers:
        if h.__class__.__name__ == "RotatingFileHandler":
            setattr(h, "maxBytes", 24 * 1024)
    lg = get_logger("spectramind.selftest.logging")
    lg.debug("Selftest logging debug — boot")
    lg.info("Selftest logging info — hello")
    lg.warning("Selftest logging warning — caution")
    lg.error("Selftest logging error — anomaly")

    # Console handler presence
    root_handlers = logging.getLogger().handlers
    has_console = any(h.__class__.__name__ == "StreamHandler" for h in root_handlers)
    if expected_console and not has_console:
        raise AssertionError("Console handler not configured but expected_console=True")
    out["checks"].append({"check": "console_handler_present", "ok": has_console})

    # File existence
    primary_log = log_dir / "spectramind.log"
    has_file = primary_log.exists() and primary_log.stat().st_size > 0 if expected_file else True
    if expected_file and not has_file:
        raise AssertionError(f"File handler expected but log not present or empty: {primary_log}")
    out["checks"].append({"check": "file_handler_wrote", "ok": has_file, "path": str(primary_log)})

    # JSONL schema
    if expected_jsonl:
        jsonl_path = log_dir / "events.jsonl"
        obj = _read_first_jsonl_line(jsonl_path)
        out["checks"].append({"check": "jsonl_schema_valid", "ok": True, "sample": obj})
    else:
        out["checks"].append({"check": "jsonl_skipped", "ok": True})

    # Rotation
    if check_rotation and expected_file:
        _, rotated = _assert_file_rotating(log_dir, base_name="spectramind.log", max_bytes=24 * 1024)
        out["checks"].append({"check": "file_rotation_happened", "ok": True, "rotated_path": str(rotated)})
    else:
        out["checks"].append({"check": "file_rotation_skipped", "ok": True})

    out["finished_at"] = datetime.now(timezone.utc).isoformat()
    out["status"] = "ok"
    return out
