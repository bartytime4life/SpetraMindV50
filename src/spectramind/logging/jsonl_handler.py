import json
from pathlib import Path
from typing import Any, Dict

class JSONLHandler:
    """Minimal JSONL writer used for logging structured events."""

    def __init__(self, filename: Path):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
