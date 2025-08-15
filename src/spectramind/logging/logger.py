from __future__ import annotations
import json, logging, logging.config, os, sys, time, platform, subprocess, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

UTC = time.gmtime


class UtcFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        # ISO 8601 UTC with milliseconds
        ts = super().formatTime(record, datefmt or "%Y-%m-%dT%H:%M:%S")
        return f"{ts}.{int(record.msecs):03d}Z"


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "file": f"{record.pathname}:{record.lineno}",
            "func": record.funcName,
            "run_id": getattr(record, "run_id", None),
            "extra": {k: v for k, v in record.__dict__.items() if k not in (
                "name","msg","args","levelname","levelno","pathname","filename","module",
                "exc_info","exc_text","stack_info","lineno","funcName","created","msecs","relativeCreated",
                "thread","threadName","processName","process","ts","taskName","message"
            )}
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# --- Run metadata capture ----------------------------------------------------
@dataclass
class RunMeta:
    run_id: str
    start_ts: float
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    git_branch: Optional[str]
    python: str
    platform: str
    machine: str
    processor: str
    env_hash: str

    @staticmethod
    def capture(run_id: str) -> "RunMeta":
        def sh(cmd):
            try:
                return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                return None

        commit = sh(["git","rev-parse","HEAD"]) or None
        status = sh(["git","status","--porcelain"]) or ""
        dirty = bool(status) if commit else None
        branch = sh(["git","rev-parse","--abbrev-ref","HEAD"]) or None
        env_dump = os.linesep.join([f"{k}={os.getenv(k)}" for k in sorted(os.environ.keys())])
        env_hash = hashlib.sha256(env_dump.encode()).hexdigest()
        return RunMeta(
            run_id=run_id,
            start_ts=time.time(),
            git_commit=commit,
            git_dirty=dirty,
            git_branch=branch,
            python=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            env_hash=env_hash,
        )


# --- Dictionary config (Hydra‑safe) -----------------------------------------

def build_logging_config(log_dir: Path, run_id: str, level: str = "INFO") -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "()": UtcFormatter,
        "format": "%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    }
    return {
        "version": 1,
        "disable_existing_loggers": False,  # Hydra adds its own; we play nice
        "filters": {
            "attach_run_id": {
                "()": "logging.Filter",
                # runtime sets run_id on record via LoggerAdapter; filter kept for completeness
            }
        },
        "formatters": {
            "human": common,
            "jsonl": {"()": JsonlFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "human",
                "stream": "ext://sys.stdout",
            },
            "file_text": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "human",
                "filename": str(log_dir / f"runtime.log"),
                "maxBytes": 10_000_000,
                "backupCount": 10,
                "encoding": "utf-8",
            },
            "file_jsonl": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "jsonl",
                "filename": str(log_dir / f"events.jsonl"),
                "maxBytes": 25_000_000,
                "backupCount": 20,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file_text", "file_jsonl"],
        },
    }


class RunLogger:
    def __init__(self, run_id: Optional[str] = None, log_dir: Optional[Path] = None, level: str = "INFO"):
        self.run_id = run_id or os.getenv("RUN_ID") or time.strftime("%Y%m%d_%H%M%S", UTC())
        base = Path(os.getenv("SPECTRA_LOG_DIR", "./logs"))
        self.log_dir = log_dir or (base / self.run_id)
        cfg = build_logging_config(self.log_dir, self.run_id, level)
        logging.config.dictConfig(cfg)
        self.logger = logging.getLogger("spectramind")
        # attach contextual adapter
        self.adapter = logging.LoggerAdapter(self.logger, extra={"run_id": self.run_id})
        # emit a run header into both sinks
        meta = RunMeta.capture(self.run_id)
        self.adapter.info("RUN_START %s", self.run_id)
        logging.getLogger("spectramind.meta").info("%s", asdict(meta))

    def get(self, name: str | None = None) -> logging.LoggerAdapter:
        return logging.LoggerAdapter(logging.getLogger(name or "spectramind"), extra={"run_id": self.run_id})


# convenience
get_logger = lambda name=None: RunLogger().get(name)

