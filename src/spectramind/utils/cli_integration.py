import datetime as dt
import json
import os
from typing import Any, Dict, Optional

from .paths import get_default_paths, ensure_dir
from .jsonl import jsonl_append


def load_run_hash_summary(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a small JSON file that summarizes run hashes/version metadata if present."""
    p = path or os.path.join(get_default_paths()["artifacts_dir"], "run_hash_summary_v50.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def append_debug_log(line: str) -> None:
    """Append a single line to v50_debug_log.md with timestamp."""
    paths = get_default_paths()
    ensure_dir(os.path.dirname(paths["debug_log_file"]))
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(paths["debug_log_file"], "a", encoding="utf-8") as f:
        f.write(f"{ts}  {line}\n")


def log_cli_invocation(command: str, config_hash: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Record a CLI invocation consistently in both the Markdown debug log and the JSONL events stream."""
    paths = get_default_paths()
    meta = load_run_hash_summary()
    line = f"[CLI] cmd='{command}' version='{meta.get('cli_version','unknown')}' cfg_hash='{config_hash or meta.get('config_hash','n/a')}'"
    append_debug_log(line)
    jsonl_append(
        paths["event_log_file"],
        {
            "time": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": "cli-invocation",
            "payload": {
                "command": command,
                "config_hash": config_hash or meta.get("config_hash"),
                "version": meta.get("cli_version", "unknown"),
                **(extra or {}),
            },
        },
    )
