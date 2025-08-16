"""
Telemetry Logger
----------------
Provides structured telemetry events for scientific reproducibility.
"""
import logging
from datetime import datetime


class TelemetryLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_event(self, event: str, metadata: dict = None) -> None:
        payload = {
            "event": event,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.logger.info(f"TelemetryEvent: {payload}")
