import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .logger import DEBUG_LOG_FILE, LOG_DIR, get_logger, log_event

VERSION_FILE_CANDIDATES = [
    Path("VERSION"),
    Path("src") / "spectramind" / "VERSION",
    Path("src") / "spectramind" / "version.txt",
]

RUN_HASH_FILE = Path("run_hash_summary_v50.json")


def _read_text_if_exists(p: Path) -> Optional[str]:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def _detect_version() -> str:
    for p in VERSION_FILE_CANDIDATES:
        s = _read_text_if_exists(p)
        if s:
            return s
    return os.environ.get("SPECTRAMIND_VERSION", "v50")


def _detect_config_hash() -> str:
    try:
        if RUN_HASH_FILE.exists():
            j = json.loads(RUN_HASH_FILE.read_text(encoding="utf-8"))
            # accept keys like {"config_hash": "..."} or {"hash": "..."}
            for k in ("config_hash", "hash", "configHash"):
                if k in j and isinstance(j[k], str) and j[k]:
                    return j[k]
    except Exception:
        pass
    return os.environ.get("SPECTRAMIND_CONFIG_HASH", "unknown")


def ensure_log_tables() -> None:
    """
    Ensure Markdown header exists for v50_debug_log.md. Safe to call anytime.
    """
    if not DEBUG_LOG_FILE.exists() or DEBUG_LOG_FILE.stat().st_size == 0:
        with open(DEBUG_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("| timestamp_utc | event | payload_json |\n")
            f.write("|---|---|---|\n")


def get_version_banner(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Prepare a standard banner payload to write at the start of every CLI call.
    """
    payload = {
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "version": _detect_version(),
        "config_hash": _detect_config_hash(),
        "pid": os.getpid(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    return payload


def write_cli_banner(
    command: str, args: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Emits a standardized banner both to the logs and to stdout for immediate visibility.
    Returns (version, config_hash) for convenience.
    """
    ensure_log_tables()
    banner = get_version_banner(extra)
    version = banner.get("version", "v50")
    config_hash = banner.get("config_hash", "unknown")

    # Console banner (always)
    logger = get_logger("spectramind")
    logger.info(
        f"SpectraMind V50 | version={version} | config_hash={config_hash} | command={command}"
    )

    # Structured event
    log_event("cli_banner", {"command": command, "args": args, **banner})

    return version, config_hash


def init_logging(level_env_default: str = "INFO") -> None:
    """
    Initialize logging with the configured level (env override allowed).
    """
    lvl = os.environ.get("SPECTRAMIND_LOG_LEVEL", level_env_default).upper()
    # Touch rotating file early for permissions visibility
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    get_logger(
        "spectramind",
        level=getattr(__import__("logging"), lvl, __import__("logging").INFO),
    )


def init_logging_for_cli(command: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    One-call initializer for CLI entrypoints:
    - init logging
    - write banner to console + logs
    - return {'version':..., 'config_hash':...} for downstream use
    """
    init_logging()
    version, cfg_hash = write_cli_banner(command=command, args=args)
    return {"version": version, "config_hash": cfg_hash}
