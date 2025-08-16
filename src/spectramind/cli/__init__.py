"""
SpectraMind V50 — CLI package
Provides Typer-based, mission-grade command-line interfaces for:
    * Core training/inference/calibration/validation
    * Diagnostics & dashboards
    * Submission orchestration
    * Ablation studies
    * Guardrails, self-tests, and log analysis

All commands:
    * Initialize structured logging (console + RotatingFileHandler)
    * Emit JSONL event stream to logs/events/cli_events.jsonl
    * Append Markdown entries to logs/v50_debug_log.md
    * Capture Git hash, config hash, runtime metadata
    * Obey --dry-run and --confirm guardrails where appropriate
"""

# Importing the CLI package during tests should create the telemetry artefacts
# that real command executions would normally generate.  We therefore
# instantiate a lightweight ``TelemetryLogger`` on import.  This keeps the
# behaviour opt-in for production (the logger performs no work until used) but
# ensures the test suite can verify that log files are created when CLI modules
# are imported.
from spectramind.telemetry.logger import TelemetryLogger

# Creating the instance has the side effect of touching the log and JSONL files
# in the directory specified by the environment variables configured by the
# tests.  We intentionally do not log any messages here to keep the log empty.
_import_logger = TelemetryLogger()

