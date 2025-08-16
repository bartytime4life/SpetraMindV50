import csv
import datetime
from pathlib import Path


class MetricsLogger:
    """
    CSV-based metrics logger.
    Columns: timestamp, step, metric_name, value
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "step", "metric_name", "value"])

    def log_metric(self, name: str, value: float, step: int) -> None:
        timestamp = datetime.datetime.utcnow().isoformat()
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, step, name, value])
