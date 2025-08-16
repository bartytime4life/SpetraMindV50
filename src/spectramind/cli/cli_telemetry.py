import functools
import os
import time
from typing import Any, Callable

from spectramind.telemetry import TelemetryManager


def _make_tm() -> TelemetryManager:
    """Create a TelemetryManager from ENV overrides (used by CLI)."""
    log_dir = os.environ.get("SPECTRAMIND_LOG_DIR", "logs/telemetry")
    enable_jsonl = os.environ.get("SPECTRAMIND_ENABLE_JSONL", "1") not in (
        "0",
        "false",
        "False",
    )
    sync_mlflow = os.environ.get("SPECTRAMIND_SYNC_MLFLOW", "0") in (
        "1",
        "true",
        "True",
    )
    sync_wandb = os.environ.get("SPECTRAMIND_SYNC_WANDB", "0") in (
        "1",
        "true",
        "True",
    )
    tm = TelemetryManager(
        log_dir=log_dir,
        enable_jsonl=enable_jsonl,
        sync_mlflow=sync_mlflow,
        sync_wandb=sync_wandb,
    )
    return tm


def with_telemetry(command_name: str) -> Callable:
    """
    Decorator to auto-log CLI command start/end/duration and config hash if provided.

    Usage:
        @with_telemetry("train")
        def train(...):
            ...
    """

    def _decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any):
            tm = _make_tm()
            start = time.time()
            config_hash = kwargs.get("config_hash") or os.environ.get(
                "SPECTRAMIND_CONFIG_HASH"
            )
            tm.log_event(
                "cli_command_start",
                {
                    "command": command_name,
                    "config_hash": config_hash,
                    "args": [repr(a) for a in args],
                    "kwargs": {k: repr(v) for k, v in kwargs.items()},
                },
            )
            try:
                result = fn(*args, **kwargs)
                duration = time.time() - start
                tm.log_event(
                    "cli_command_end",
                    {
                        "command": command_name,
                        "status": "ok",
                        "duration_sec": round(duration, 4),
                    },
                )
                return result
            except Exception as e:  # pragma: no cover - defensive
                duration = time.time() - start
                tm.log_event(
                    "cli_command_end",
                    {
                        "command": command_name,
                        "status": "error",
                        "duration_sec": round(duration, 4),
                        "error": repr(e),
                    },
                )
                raise

        return _wrapped

    return _decorator
