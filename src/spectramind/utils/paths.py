import os
from typing import Dict, Optional

V50_DEFAULTS = {
    "root_markers": [".git", ".spectramind", "pyproject.toml", "README.md"],
    "logs_dir": "logs",
    "artifacts_dir": "artifacts",
    "reports_dir": "reports",
    "run_info_dir": "artifacts/run_info",
    "debug_log_file": "v50_debug_log.md",
    "jsonl_log_file": "logs/events.jsonl",
    "text_log_file": "logs/spectramind.log",
    "event_log_file": "logs/events.jsonl",  # alias to JSONL log
}


def ensure_dir(path: str) -> str:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def find_repo_root(start: Optional[str] = None) -> str:
    """Find the repository root by walking up until we see any known root marker."""
    cur = os.path.abspath(start or os.getcwd())
    while True:
        for marker in V50_DEFAULTS["root_markers"]:
            if os.path.exists(os.path.join(cur, marker)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # Fallback to cwd if no markers found (still deterministic)
            return os.path.abspath(start or os.getcwd())
        cur = parent


def repo_relpath(path: str) -> str:
    root = find_repo_root()
    try:
        return os.path.relpath(os.path.abspath(path), root)
    except Exception:
        return os.path.abspath(path)


def get_default_paths() -> Dict[str, str]:
    """Compute and return canonical paths for logs, artifacts, reports, and debug logs."""
    root = find_repo_root()
    logs_dir = ensure_dir(os.path.join(root, V50_DEFAULTS["logs_dir"]))
    artifacts_dir = ensure_dir(os.path.join(root, V50_DEFAULTS["artifacts_dir"]))
    reports_dir = ensure_dir(os.path.join(root, V50_DEFAULTS["reports_dir"]))
    run_info_dir = ensure_dir(os.path.join(root, V50_DEFAULTS["run_info_dir"]))
    return {
        "root": root,
        "logs_dir": logs_dir,
        "artifacts_dir": artifacts_dir,
        "reports_dir": reports_dir,
        "run_info_dir": run_info_dir,
        "log_file": os.path.join(root, V50_DEFAULTS["text_log_file"]),
        "jsonl_log_file": os.path.join(root, V50_DEFAULTS["jsonl_log_file"]),
        "event_log_file": os.path.join(root, V50_DEFAULTS["event_log_file"]),
        "debug_log_file": os.path.join(root, V50_DEFAULTS["debug_log_file"]),
    }
