#!/usr/bin/env bash

# install_bin_spectramind.sh
# Create SpectraMind V50 CLI wrappers in bin/

set -euo pipefail

find_repo_root() {
  if command -v git >/dev/null 2>&1; then
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
      git rev-parse --show-toplevel
      return
    fi
  fi
  local start="$(pwd)"
  while true; do
    if [ -d .git ] || [ -f pyproject.toml ] || [ -f spectramind.py ] || [ -d src ]; then
      pwd
      return
    fi
    if [ "$(pwd)" = "/" ]; then
      echo "$start"
      return
    fi
    cd ..
  done
}

ROOT="$(find_repo_root)"
cd "$ROOT"
mkdir -p bin

cat > bin/wrapper_base.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for SpectraMind V50 CLI wrappers."""
import os, sys, json, time, subprocess, hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

def find_repo_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    p = start.resolve()
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except Exception:
        pass
    while True:
        if (p/".git").exists() or (p/"pyproject.toml").exists() or (p/"spectramind.py").exists() or (p/"src").exists():
            return p
        if p.parent == p:
            return start
        p = p.parent

def repo_git_info(root: Path) -> dict:
    info = {"branch": None, "commit": None, "dirty": None}
    try:
        b = subprocess.run(["git","rev-parse","--abbrev-ref","HEAD"], cwd=root, capture_output=True, text=True, check=True)
        c = subprocess.run(["git","rev-parse","HEAD"], cwd=root, capture_output=True, text=True, check=True)
        d = subprocess.run(["git","status","--porcelain"], cwd=root, capture_output=True, text=True, check=True)
        info["branch"] = b.stdout.strip()
        info["commit"] = c.stdout.strip()
        info["dirty"] = bool(d.stdout.strip())
    except Exception:
        pass
    return info

def read_config_hash(root: Path) -> Optional[str]:
    for p in [root/"run_hash_summary_v50.json", root/"configs"/"run_hash_summary_v50.json", root/"artifacts"/"run_hash_summary_v50.json"]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k in ("config_hash","hash","run_hash","configHash","runHash"):
                    if k in data and data[k]:
                        return str(data[k])
            except Exception:
                pass
    return None

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_markdown_log(root: Path, content: str) -> None:
    md = root/"v50_debug_log.md"
    ensure_parent(md)
    with md.open("a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")

def append_events_jsonl(root: Path, event: dict) -> None:
    jl = root/"events.jsonl"
    ensure_parent(jl)
    with jl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False))
        f.write("\n")

def which_python() -> str:
    return sys.executable or "python3"

def detect_invocation(root: Path, candidates: List[Tuple[str,List[str]]], file_candidates: List[Path]):
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
    src = str((root/"src").resolve())
    if "PYTHONPATH" in env and src not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src
    if env_overrides:
        env.update(env_overrides)
    try:
        return subprocess.run(cmd_argv, env=env, cwd=str(cwd) if cwd else None).returncode
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
        dur = round(time.time()-t0, 3)
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
        append_events_jsonl(root, {
            "ts": ts,
            "tool": tool_name,
            "event": "resolve_error",
            "cmd": cmd,
            "repo": str(root),
            "git": git,
            "config_hash": cfg_hash,
            "duration_s": dur,
            "error": "resolve_failed",
        })
        return 2
    full_cmd = resolved + args
    cmd_str = " ".join(full_cmd)
    rc = run_forwarded(full_cmd, cwd=root)
    dur = round(time.time()-t0, 3)
    md = (
        f"\n### {ts} — {tool_name} RC={rc}\n"
        f"- cmd: `{cmd_str}`\n"
        f"- repo: `{root}`\n"
        f"- git: branch={git.get('branch')} commit={git.get('commit')} dirty={git.get('dirty')}\n"
        f"- config_hash: `{cfg_hash}`\n"
        f"- duration_s: {dur}\n"
    )
    append_markdown_log(root, md)
    append_events_jsonl(root, {
        "ts": ts,
        "tool": tool_name,
        "event": "run",
        "cmd_argv": full_cmd,
        "repo": str(root),
        "git": git,
        "config_hash": cfg_hash,
        "duration_s": dur,
        "rc": rc,
        "env": {"python": sys.version.split()[0], "hydra_full_error": True}
    })
    return rc
PY
chmod +x bin/wrapper_base.py

write_wrapper() {
  local name="$1"; shift
  local modules="$1"; shift
  local files="$1"; shift
  cat > "bin/$name" <<PY
#!/usr/bin/env python3
import sys
from pathlib import Path
from wrapper_base import wrapper_main, find_repo_root

if __name__ == "__main__":
    root = find_repo_root()
    module_candidates = $modules
    file_candidates = [$(echo "$files")]
    sys.exit(wrapper_main("$name", module_candidates, file_candidates, sys.argv[1:]))
PY
  chmod +x "bin/$name"
}

write_wrapper spectramind "[('-m',['spectramind'])]" "root / 'spectramind.py'"
write_wrapper train_v50 "[('-m',['spectramind.training.train_v50']), ('-m',['train_v50'])]" "root/'src'/'spectramind'/'training'/'train_v50.py', root/'train_v50.py'"
write_wrapper predict_v50 "[('-m',['spectramind.inference.predict_v50']), ('-m',['predict_v50'])]" "root/'src'/'spectramind'/'inference'/'predict_v50.py', root/'predict_v50.py'"
write_wrapper calibrate_v50 "[('-m',['spectramind.calibration.pipeline']), ('-m',['calibration.pipeline']), ('-m',['cli_calibrate'])]" "root/'src'/'spectramind'/'calibration'/'pipeline.py', root/'cli_calibrate.py'"
write_wrapper diagnose_v50 "[('-m',['cli_diagnose']), ('-m',['spectramind.cli_diagnose'])]" "root/'cli_diagnose.py', root/'src'/'spectramind'/'cli_diagnose.py'"
write_wrapper ablate_v50 "[('-m',['cli_ablate']), ('-m',['spectramind.cli_ablate'])]" "root/'cli_ablate.py', root/'src'/'spectramind'/'cli_ablate.py', root/'auto_ablate_v50.py', root/'src'/'spectramind'/'auto_ablate_v50.py'"
write_wrapper selftest_v50 "[('-m',['selftest']), ('-m',['spectramind.tools.selftest'])]" "root/'selftest.py', root/'tools'/'selftest.py', root/'src'/'spectramind'/'tools'/'selftest.py'"
write_wrapper generate_html_report "[('-m',['generate_html_report']), ('-m',['spectramind.diagnostics.generate_html_report'])]" "root/'generate_html_report.py', root/'src'/'spectramind'/'diagnostics'/'generate_html_report.py', root/'tools'/'generate_html_report.py'"

cat > bin/run_pipeline_v50 <<'PY'
#!/usr/bin/env python3
"""Run SpectraMind V50 pipeline: train -> predict -> calibrate -> diagnose -> report."""
import sys, shlex, subprocess, os, time
from pathlib import Path
from wrapper_base import find_repo_root, append_markdown_log, append_events_jsonl, now_iso

STAGES = [
    ("train_v50", "--no-train"),
    ("predict_v50", "--no-predict"),
    ("calibrate_v50", "--no-calibrate"),
    ("diagnose_v50", "--no-diagnose"),
    ("generate_html_report", "--no-report"),
]

def run_bin(root: Path, name: str, argstr: str) -> int:
    exe = root / 'bin' / name
    if not exe.exists():
        sys.stderr.write(f"[run_pipeline_v50] missing {exe}\n")
        return 2
    cmd = [str(exe)] + (shlex.split(argstr) if argstr else [])
    env = os.environ.copy()
    env.setdefault('HYDRA_FULL_ERROR', '1')
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, cwd=str(root)).returncode
    dur = round(time.time() - t0, 3)
    append_events_jsonl(root, {"ts": now_iso(), "tool": "run_pipeline_v50", "stage": name, "cmd": cmd, "rc": rc, "duration_s": dur})
    return rc

def main(argv):
    root = find_repo_root()
    flags = {name: True for name, _ in STAGES}
    arg_map = {name: "" for name, _ in STAGES}
    for a in argv:
        handled = False
        for name, disable_flag in STAGES:
            if a == disable_flag:
                flags[name] = False
                handled = True
                break
            prefix = f"--{name.replace('_','-')}-args="
            if a.startswith(prefix):
                arg_map[name] = a.split("=", 1)[1]
                handled = True
                break
        if not handled:
            sys.stderr.write(f"[run_pipeline_v50] Unknown arg: {a}\n")
    ts = now_iso()
    stages = [n for n in flags if flags[n]]
    append_markdown_log(root, f"\n### {ts} — run_pipeline_v50 starting\n- stages: {', '.join(stages)}\n- root: `{root}`\n")
    for name in stages:
        rc = run_bin(root, name, arg_map[name])
        if rc != 0:
            append_markdown_log(root, f"- stage `{name}` FAILED rc={rc} — aborting pipeline\n")
            return rc
    append_markdown_log(root, "- pipeline SUCCESS — all stages completed\n")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
PY
chmod +x bin/run_pipeline_v50

echo "bin/ CLI wrappers installed at: $ROOT/bin"
