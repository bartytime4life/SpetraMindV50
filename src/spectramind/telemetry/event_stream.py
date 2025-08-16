import json
from pathlib import Path
from typing import Any, Dict, Optional


class EventStream:
    """
    JSONL event stream for telemetry.
    Each event is written as a single JSON object on a line.
    """

    def __init__(self, jsonl_path: Optional[Path | str] = None) -> None:
        self.jsonl_file = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_file:
            self.jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    def write_event(self, event: Dict[str, Any]) -> None:
        if self.jsonl_file is None:
            return
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
