from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json, logging

log = logging.getLogger("spectramind.events")


@dataclass
class Event:
    kind: str
    phase: str  # e.g., "start" | "end" | "step"
    step: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def emit(self, logger: logging.LoggerAdapter):
        logger.info(json.dumps(asdict(self)))

