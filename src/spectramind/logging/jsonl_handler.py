"""
JSONL Event Stream Handler
--------------------------
Logs each record as a structured JSON line for downstream analysis.
"""
import logging
import json
from pathlib import Path
from datetime import datetime


class JSONLHandler(logging.Handler):
    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            self.filepath.write(json.dumps(entry) + "\n")
            self.filepath.flush()
        except Exception:
            self.handleError(record)
