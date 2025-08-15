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
