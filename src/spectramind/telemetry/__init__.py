"""
SpectraMind V50 - Telemetry Package

This package provides unified telemetry management for logging, event streaming,
metrics capture, and diagnostics hooks. It integrates with the SpectraMind CLI,
Hydra configs, and diagnostics dashboards to ensure full reproducibility and
mission-grade telemetry handling.
"""

from .diagnostics_hooks import DiagnosticsHooks
from .event_stream import EventStream
from .metrics_logger import MetricsLogger
from .telemetry_manager import TelemetryManager
from .logger import TelemetryLogger, get_telemetry_logger, get_logger

__all__ = [
    "TelemetryManager",
    "EventStream",
    "MetricsLogger",
    "DiagnosticsHooks",
    "TelemetryLogger",
    "get_telemetry_logger",
    "get_logger",
]
