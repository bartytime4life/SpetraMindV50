import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JSONLStreamLogger:
    """
    Lightweight JSONL writer for analytics ingestion and dashboards.
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(exist_ok=True, parents=True)

    def write(self, event_type: str, data: Dict[str, Any]) -> None:
        record = {"ts": datetime.utcnow().isoformat(), "type": event_type, "data": data}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
