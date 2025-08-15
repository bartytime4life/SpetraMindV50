from __future__ import annotations

import json
import os
from typing import Optional

import typer

from ..conf_helpers.hashing import config_hash
from ..train.trainer import train_v50
from ..utils.git_env import capture_git_env
from ..utils.hydra_safe import load_yaml
from ..utils.logging_setup import build_logger, log_event

app = typer.Typer(help="SpectraMind V50 — Unified CLI")


@app.command()
def version() -> None:
    logger, jsonl, _ = build_logger()
    meta = capture_git_env()
    log_event(jsonl, "cli.version", {"meta": meta})
    print(json.dumps({"version": "v50-cli", **meta}, indent=2))


@app.command()
def train(config: str = typer.Option(..., help="Path to config_v50.yaml")) -> None:
    cfg = load_yaml(config)
    cfg["_config_hash"] = config_hash(cfg)
    out = train_v50(cfg)
    print(json.dumps({"status": "ok", **out}, indent=2))


@app.command()
def analyze_log(
    log_jsonl: str = typer.Option(
        "logs/spectramind_v50.jsonl", help="Path to JSONL event stream"
    ),
    out_md: Optional[str] = typer.Option(None, help="Optional Markdown summary path"),
) -> None:
    from collections import Counter

    n = 0
    kinds = Counter()
    if not os.path.exists(log_jsonl):
        print(json.dumps({"error": "missing log file", "path": log_jsonl}, indent=2))
        raise typer.Exit(code=1)
    with open(log_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = (
                    json.loads(line.split(" | ", 3)[-1])
                    if " | " in line
                    else json.loads(line)
                )
                kinds[ev.get("kind", "unknown")] += 1
                n += 1
            except Exception:
                pass
    summary = {"total_events": n, "by_kind": kinds.most_common()}
    print(json.dumps(summary, indent=2))
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as fh:
            fh.write("# SpectraMind V50 — Log Summary\n\n")
            fh.write(f"- Total events: {n}\n\n")
            fh.write("| Kind | Count |\n|---|---:|\n")
            for k, c in kinds.most_common():
                fh.write(f"| {k} | {c} |\n")


if __name__ == "__main__":
    app()
