import datetime as _dt
from typing import Any, Dict, Optional

from .jsonl import jsonl_append


class EventStream:
    """Lightweight JSONL event emitter with a stable schema (time, event, payload)."""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event with a UTC timestamp to the JSONL stream."""
        jsonl_append(
            self.jsonl_path,
            {
                "time": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "event": event,
                "payload": payload or {},
            },
        )
