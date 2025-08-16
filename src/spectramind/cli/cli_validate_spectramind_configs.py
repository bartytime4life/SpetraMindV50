#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpectraMind V50 — CLI to validate configs/model/*.yaml."""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import OmegaConf

from ..config.schemas_spectramind import load_and_validate_yaml

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Validate SpectraMind model YAMLs.",
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _append_md_log(msg: str, repo_root: Path) -> None:
    md = repo_root / "v50_debug_log.md"
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    line = f"- [{timestamp}] cli_validate_spectramind_configs: {msg}\n"
    try:
        with md.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:  # pragma: no cover - logging failure should not crash
        print(f"[log-warn] failed to append to v50_debug_log.md: {e}", file=sys.stderr)


def _append_jsonl_event(event: dict, repo_root: Path) -> None:
    ev = repo_root / "events.jsonl"
    try:
        with ev.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        print(f"[log-warn] failed to append to events.jsonl: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core command
# ---------------------------------------------------------------------------


@app.command("validate")
def validate(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        exists=False,
        readable=True,
        help="Path to a single YAML (default: validate all under configs/model).",
    ),
    validate_all: bool = typer.Option(
        False,
        "--all",
        help="Validate all YAMLs under configs/model.",
    ),
    emit_merged: Optional[Path] = typer.Option(
        None,
        "--emit-merged",
        help="If set, write a single merged validated YAML (spectramind root) to this path.",
    ),
) -> None:
    """Validate one or more YAML files against the structured schemas."""
    repo_root = Path(os.environ.get("SPECTRAMIND_REPO_ROOT", Path.cwd())).resolve()
    cfg_dir = repo_root / "configs" / "model"

    targets: List[Path] = []
    if path is not None:
        targets = [path]
    elif validate_all:
        if not cfg_dir.exists():
            typer.echo(f"[error] directory missing: {cfg_dir}", err=True)
            raise typer.Exit(code=2)
        targets = sorted(
            p for p in cfg_dir.glob("*.yaml") if p.name != "__init__.yaml"
        )
    else:
        typer.echo("Specify --path FILE or --all", err=True)
        raise typer.Exit(code=2)

    ok = True
    merged_cfgs = []

    for p in targets:
        try:
            cfg = load_and_validate_yaml(str(p))
            merged_cfgs.append(cfg)
            typer.echo(f"[ok] {p}")
            _append_md_log(f"validated: {p}", repo_root)
            _append_jsonl_event(
                {
                    "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "event": "validate_spectramind_yaml",
                    "path": str(p),
                    "status": "ok",
                },
                repo_root,
            )
        except Exception as e:  # pragma: no cover - CLI failure paths
            ok = False
            typer.echo(f"[fail] {p} :: {e}", err=True)
            _append_md_log(f"validate-fail: {p} :: {e}", repo_root)
            _append_jsonl_event(
                {
                    "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "event": "validate_spectramind_yaml",
                    "path": str(p),
                    "status": "fail",
                    "error": str(e),
                },
                repo_root,
            )

    if emit_merged and ok:
        final_cfg = merged_cfgs[-1] if merged_cfgs else None
        if final_cfg is not None:
            emit_merged.parent.mkdir(parents=True, exist_ok=True)
            with open(emit_merged, "w", encoding="utf-8") as f:
                f.write(OmegaConf.to_yaml(final_cfg, resolve=True))
            typer.echo(f"[write] merged YAML -> {emit_merged}")
            _append_md_log(f"write-merged: {emit_merged}", repo_root)
            _append_jsonl_event(
                {
                    "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "event": "write_merged_yaml",
                    "path": str(emit_merged),
                    "status": "ok",
                },
                repo_root,
            )

    raise typer.Exit(code=0 if ok else 1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
