# Copyright (c) 2025
# SpectraMind V50 Logging Package
# Provides mission-grade logging utilities:
# - Structured console + rotating file logging
# - Markdown + JSONL event streams
# - MLflow optional sync
# - Git/ENV/Hydra capture
# - CLI auto-instrumentation (Typer)
from .bootstrap import (
    ensure_log_tables,
    get_version_banner,
    init_logging,
    init_logging_for_cli,
    write_cli_banner,
)
from .git_env_capture import capture_git_env_state
from .jsonl_stream import JSONLStreamLogger
from .logger import (
    DEBUG_LOG_FILE,
    LOG_DIR,
    get_logger,
    log_cli_call,
    log_event,
    setup_rotating_handlers,
)
from .mlflow_sync import MLflowSync

__all__ = [
    "get_logger",
    "log_event",
    "log_cli_call",
    "setup_rotating_handlers",
    "LOG_DIR",
    "DEBUG_LOG_FILE",
    "JSONLStreamLogger",
    "MLflowSync",
    "capture_git_env_state",
    "init_logging",
    "init_logging_for_cli",
    "get_version_banner",
    "ensure_log_tables",
    "write_cli_banner",
]
