from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


class HumanFileHandler(RotatingFileHandler):
    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        msg = self.format(record)
        try:
            with open(self.baseFilename, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def setup_logging(log_path: str = "v50_debug_log.md", jsonl_path: str = "events.jsonl") -> logging.Logger:
    logger = logging.getLogger("spectramind")
    logger.setLevel(logging.INFO)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    hf = HumanFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    hf.setFormatter(logging.Formatter("%(asctime)sZ [%(levelname)s] %(message)s"))
    logger.addHandler(hf)

    class JSONL(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            try:
                event: Dict[str, Any] = {
                    "ts": record.asctime if hasattr(record, "asctime") else None,
                    "level": record.levelname,
                    "msg": record.getMessage(),
                }
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass

    logger.addHandler(JSONL())
    return logger
