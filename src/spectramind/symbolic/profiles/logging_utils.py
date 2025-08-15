"""Mission-grade logging utilities: console + RotatingFileHandler + JSONL events, v50_debug_log
appends, ENV/Git snapshot capture. Keeps dependencies to stdlib only."""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def setup_logging(
    name: str = "profiles",
    repo_root: Optional[Path] = None,
    log_dir_rel: str = "logs",
    log_file_name: str = "spectramind_profiles.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure mission-grade logging with console + rotating file.
    Returns a configured logger."""
    if repo_root is None:
        repo_root = Path.cwd()
    log_dir = repo_root / log_dir_rel
    _ensure_dir(log_dir)
    log_path = log_dir / log_file_name

    logger = logging.getLogger(f"spectramind.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(str(log_path), maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    logger.info("Logging initialized")
    return logger


def write_jsonl_event(
    repo_root: Path,
    event: Dict[str, Any],
    jsonl_rel_path: str = "logs/events.jsonl",
) -> None:
    """Append an event (dict) to JSONL event stream, creating file/dirs if missing."""
    path = repo_root / jsonl_rel_path
    _ensure_dir(path.parent)
    event = dict(event)  # shallow copy
    event.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def append_v50_debug_log(
    repo_root: Path,
    message: str,
    md_rel_path: str = "v50_debug_log.md",
) -> None:
    """Append a single line to v50_debug_log.md with timestamp."""
    path = repo_root / md_rel_path
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"- [{ts}] {message}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def git_snapshot(repo_root: Path) -> Dict[str, Any]:
    """Capture lightweight git snapshot for reproducibility logging. Non-fatal if git not present."""
    snap: Dict[str, Any] = {"git_present": False}
    try:
        root_out = _run(["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"])
        if root_out.returncode != 0:
            return snap
        snap["git_present"] = True
        snap["root"] = root_out.stdout.strip()
        for k, args in [
            ("head", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain=v1"]),
            ("branch", ["branch", "--show-current"]),
        ]:
            cp = _run(["git", "-C", str(repo_root)] + args)
            snap[k] = cp.stdout.strip()
        lg = _run(["git", "-C", str(repo_root), "log", "--oneline", "-n", "5"])
        snap["last5"] = lg.stdout.strip().splitlines()
    except Exception as e:  # pragma: no cover - best effort
        snap["error"] = repr(e)
    return snap


def env_snapshot() -> Dict[str, Any]:
    """Minimal environment snapshot for diagnostics."""
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
