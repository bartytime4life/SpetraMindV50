import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# --- Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
EVENTS_DIR = LOG_DIR / "events"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIAG_DIR = PROJECT_ROOT / "diagnostics"
SRC_DIR = PROJECT_ROOT / "src"
PKG_DIR = SRC_DIR / "spectramind"
CLI_DIR = PKG_DIR / "cli"
DEFAULT_EVENT_JSONL = EVENTS_DIR / "cli_events.jsonl"
MD_LOG = LOG_DIR / "v50_debug_log.md"
ROTATING_LOG = LOG_DIR / "cli.log"
ROTATE_BYTES = 10 * 1024 * 1024
ROTATE_BACKUPS = 5
CLI_VERSION = "v50.2025.08.15"

# --- FS ensure
for p in [LOG_DIR, EVENTS_DIR, REPORTS_DIR, DIAG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- Logging setup
_LOGGER: Optional[logging.Logger] = None

def init_logging(name: str = "spectramind.cli", level: int = logging.INFO) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    # Rotating file
    fh = RotatingFileHandler(ROTATING_LOG, maxBytes=ROTATE_BYTES, backupCount=ROTATE_BACKUPS)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
    _LOGGER = logger
    return logger


logger = init_logging()

# --- Utilities

def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def git_info() -> Dict[str, str]:
    info = {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        info["git_commit"] = commit
    except Exception:
        pass
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        info["git_branch"] = branch
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode()
        info["git_dirty"] = "dirty" if status.strip() else "clean"
    except Exception:
        pass
    return info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def config_hash(paths: Iterable[Union[str, Path]]) -> str:
    parts: List[str] = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            parts.append(f"{p.name}:{sha256_file(p)}")
        elif p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file():
                    parts.append(f"{sub.relative_to(p)}:{sha256_file(sub)}")
        else:
            parts.append(f"{p}:missing")
    digest = sha256_string("|".join(parts)) if parts else sha256_string("none")
    return digest[:12]


def write_jsonl_event(event_type: str, payload: Dict[str, Any], jsonl_path: Path = DEFAULT_EVENT_JSONL) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_markdown_log(
    command: str,
    args: List[str],
    cli_version: str,
    cfg_hash: str,
    status: str,
    started_at: float,
    finished_at: float,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    gi = git_info()
    host = platform.node()
    md = []
    md.append(f"### {datetime.now().isoformat()} — {command} status: {status}")
    md.append("")
    md.append(f"- CLI Version: {cli_version}")
    md.append(f"- Host: {host}")
    md.append(f"- Git: {gi.get('git_branch','?')} @ {gi.get('git_commit','?')} ({gi.get('git_dirty','?')})")
    md.append(f"- Config Hash: {cfg_hash}")
    md.append(f"- Args: {' '.join(args)}")
    md.append(f"- Duration: {finished_at - started_at:.2f}s")
    if extras:
        for k, v in extras.items():
            md.append(f"- {k}: {v}")
    md.append("")
    md_log = "\n".join(md) + "\n"
    with MD_LOG.open("a", encoding="utf-8") as f:
        f.write(md_log)


def capture_env_snapshot(out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    snaps: Dict[str, str] = {}
    # pip freeze
    try:
        pf = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        (out_dir / "pip-freeze.txt").write_text(pf, encoding="utf-8")
        snaps["pip_freeze"] = str((out_dir / "pip-freeze.txt").relative_to(PROJECT_ROOT))
    except Exception:
        pass
    # conda
    try:
        ci = subprocess.check_output(["conda", "env", "export"]).decode()
        (out_dir / "conda-env.yaml").write_text(ci, encoding="utf-8")
        snaps["conda_env"] = str((out_dir / "conda-env.yaml").relative_to(PROJECT_ROOT))
    except Exception:
        pass
    # nvidia-smi
    try:
        smi = subprocess.check_output(["nvidia-smi"]).decode()
        (out_dir / "nvidia-smi.txt").write_text(smi, encoding="utf-8")
        snaps["nvidia_smi"] = str((out_dir / "nvidia-smi.txt").relative_to(PROJECT_ROOT))
    except Exception:
        pass
    # python env
    (out_dir / "python.txt").write_text(sys.version, encoding="utf-8")
    snaps["python"] = str((out_dir / "python.txt").relative_to(PROJECT_ROOT))
    return snaps


def hydra_kv_to_cli(overrides: List[str]) -> List[str]:
    # Normalize Hydra-like overrides (key=value) preserving pass-through CLI tokens
    out: List[str] = []
    for o in overrides:
        if "=" in o and not o.strip().startswith("--"):
            out.append(f"+{o}")
        else:
            out.append(o)
    return out


def open_in_browser(path: Union[str, Path]) -> None:
    try:
        import webbrowser
        webbrowser.open_new_tab(str(path))
    except Exception:
        logger.warning("Failed to open browser for %s", path)


def call_python_module(mod: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", mod] + args
    logger.info("Exec: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def call_python_file(py_file: Path, args: List[str]) -> int:
    cmd = [sys.executable, str(py_file)] + args
    logger.info("Exec: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def find_module_or_script(module: str, candidates: List[Path]) -> Tuple[str, Optional[Path]]:
    """
    Returns ("module", None) if importable, else ("script", path) if file exists, else ("missing", None)
    """
    try:
        import importlib
        importlib.import_module(module)
        return ("module", None)
    except Exception:
        for c in candidates:
            if c.exists():
                return ("script", c)
        return ("missing", None)


@contextmanager
def command_session(command: str, args: List[str], cfg_paths: Optional[List[Union[str, Path]]] = None):
    started = time.time()
    cfg_hash = config_hash(cfg_paths or [])
    write_jsonl_event("cli_start", {
        "command": command,
        "args": args,
        "cli_version": CLI_VERSION,
        "config_hash": cfg_hash,
        "git": git_info(),
    })
    status = "success"
    err: Optional[str] = None
    try:
        yield cfg_hash, started
    except SystemExit as se:  # propagate but log
        status = f"system_exit({getattr(se, 'code', 0)})"
        err = None
        raise
    except Exception as e:
        status = "error"
        err = "".join(traceback.format_exception(e))
        raise
    finally:
        finished = time.time()
        extras = {"error": err[:2000]} if err else None
        append_markdown_log(command, args, CLI_VERSION, cfg_hash, status, started, finished, extras=extras)
        write_jsonl_event("cli_finish", {
            "command": command,
            "status": status,
            "duration_sec": finished - started,
            "config_hash": cfg_hash,
        })


def ensure_tools():
    # Helpful runtime checks
    try:
        import typer  # noqa: F401
    except Exception as e:
        logger.error("Typer not installed. pip install typer[all]  (%s)", e)
        sys.exit(2)


def md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt_line(r: List[str]) -> str:
        return "| " + " | ".join((c + " " * (widths[i] - len(c))) for i, c in enumerate(r)) + " |"

    lines = [fmt_line(rows[0]), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines += [fmt_line(r) for r in rows[1:]]
    return "\n".join(lines)
