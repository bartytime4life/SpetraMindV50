#!/usr/bin/env python3
"""Common utilities for SpectraMind V50 CLI wrappers."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    p = start.resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        )
        return Path(out.stdout.strip())
    except Exception:
        pass
    while True:
        if (
            (p / ".git").exists()
            or (p / "pyproject.toml").exists()
            or (p / "spectramind.py").exists()
            or (p / "src").exists()
        ):
            return p
        if p.parent == p:
            return start
        p = p.parent


def repo_git_info(root: Path) -> dict:
    info = {"branch": None, "commit": None, "dirty": None}
    try:
        b = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        c = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        d = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        )
        info["branch"] = b.stdout.strip()
        info["commit"] = c.stdout.strip()
        info["dirty"] = bool(d.stdout.strip())
    except Exception:
        pass
    return info


def read_config_hash(root: Path) -> str | None:
    for p in [
        root / "run_hash_summary_v50.json",
        root / "configs" / "run_hash_summary_v50.json",
        root / "artifacts" / "run_hash_summary_v50.json",
    ]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k in ("config_hash", "hash", "run_hash", "configHash", "runHash"):
                    if data.get(k):
                        return str(data[k])
            except Exception:
                pass
    return None


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def append_markdown_log(root: Path, content: str) -> None:
    md = root / "v50_debug_log.md"
    ensure_parent(md)
    with md.open("a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")


def append_events_jsonl(root: Path, event: dict) -> None:
    jl = root / "events.jsonl"
    ensure_parent(jl)
    with jl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False))
        f.write("\n")


def which_python() -> str:
    return sys.executable or "python3"


def detect_invocation(
    root: Path, candidates: list[tuple[str, list[str]]], file_candidates: list[Path]
):
    py = which_python()
    for kind, args in candidates:
        if kind == "-m":
            try:
                __import__(args[0])
                return [py, "-m", *args]
            except Exception:
                pass
        elif kind == "file":
            for fp in file_candidates:
                if fp.exists():
                    return [py, str(fp), *args]
    for kind, args in candidates:
        if kind == "-m":
            return [py, "-m", *args]
    return None


def run_forwarded(cmd_argv, env_overrides=None, cwd=None) -> int:
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    root = find_repo_root()
    src = str((root / "src").resolve())
    if "PYTHONPATH" in env and src not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src
    if env_overrides:
        env.update(env_overrides)
    try:
        return subprocess.run(
            cmd_argv, env=env, cwd=str(cwd) if cwd else None, check=False
        ).returncode
    except FileNotFoundError:
        return 127
    except KeyboardInterrupt:
        return 130


def wrapper_main(tool_name: str, module_candidates, file_candidates, passthrough_args=None) -> int:
    root = find_repo_root()
    git = repo_git_info(root)
    cfg_hash = read_config_hash(root)
    t0 = time.time()
    ts = now_iso()
    args = [] if passthrough_args is None else list(passthrough_args)
    resolved = detect_invocation(root, module_candidates, file_candidates)
    if resolved is None:
        msg = f"[{tool_name}] Could not locate module or script. Tried modules={module_candidates} files={file_candidates}"
        sys.stderr.write(msg + "\n")
        dur = round(time.time() - t0, 3)
        cmd = f"{tool_name} " + " ".join(args)
        md = (
            f"\n### {ts} — {tool_name} FAILED (resolve)\n"
            f"- cmd: `{cmd}`\n"
            f"- repo: `{root}`\n"
            f"- git: branch={git.get('branch')} commit={git.get('commit')} dirty={git.get('dirty')}\n"
            f"- config_hash: `{cfg_hash}`\n"
            f"- duration_s: {dur}\n"
            f"- error: Could not resolve underlying module or file.\n"
        )
        append_markdown_log(root, md)
        append_events_jsonl(
            root,
            {
                "ts": ts,
                "tool": tool_name,
                "event": "resolve_error",
                "cmd": cmd,
                "repo": str(root),
                "git": git,
                "config_hash": cfg_hash,
                "duration_s": dur,
                "error": "resolve_failed",
            },
        )
        return 2
    full_cmd = resolved + args
    cmd_str = " ".join(full_cmd)
    rc = run_forwarded(full_cmd, cwd=root)
    dur = round(time.time() - t0, 3)
    md = (
        f"\n### {ts} — {tool_name} RC={rc}\n"
        f"- cmd: `{cmd_str}`\n"
        f"- repo: `{root}`\n"
        f"- git: branch={git.get('branch')} commit={git.get('commit')} dirty={git.get('dirty')}\n"
        f"- config_hash: `{cfg_hash}`\n"
        f"- duration_s: {dur}\n"
    )
    append_markdown_log(root, md)
    append_events_jsonl(
        root,
        {
            "ts": ts,
            "tool": tool_name,
            "event": "run",
            "cmd_argv": full_cmd,
            "repo": str(root),
            "git": git,
            "config_hash": cfg_hash,
            "duration_s": dur,
            "rc": rc,
            "env": {"python": sys.version.split()[0], "hydra_full_error": True},
        },
    )
    return rc
