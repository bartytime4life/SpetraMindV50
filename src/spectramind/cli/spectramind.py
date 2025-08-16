import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .common import (
    CLI_VERSION,
    PROJECT_ROOT,
    LOG_DIR,
    EVENTS_DIR,
    REPORTS_DIR,
    logger,
    command_session,
    git_info,
    config_hash,
    md_table,
)
from . import selftest as selftest_mod
from . import cli_core_v50 as core
from . import cli_diagnose as diag
from . import cli_submit as submit
from . import cli_ablate as ablate
from . import cli_bundle as bundle
from . import cli_dashboard_mini as dashmini

app = typer.Typer(add_completion=True, no_args_is_help=True, help="SpectraMind V50 — Unified CLI")


@app.callback()
def cli_root(ctx: typer.Context, runtime: str = typer.Option("default", "--runtime", help="Runtime environment config")):
    """Root SpectraMind CLI with global options."""
    ctx.obj = ctx.obj or {}
    ctx.obj["runtime"] = runtime


# Register sub-apps
app.add_typer(core.app, name="core", help="Core train/predict/calibrate/validate/explain")
app.add_typer(diag.app, name="diagnose", help="Diagnostics and dashboard")
app.add_typer(submit.app, name="submit", help="Submission orchestration")
app.add_typer(ablate.app, name="ablate", help="Symbolic-aware ablations")
app.add_typer(bundle.app, name="bundle", help="Packaging/bundling")
app.add_typer(dashmini.app, name="dashboard-mini", help="Quick report builder")


@app.command("version")
def version(
    cfg: Optional[str] = typer.Option(None, "--config", "-c", help="Optional config file to hash"),
):
    """Print CLI version, git info, and optional config hash; log to v50_debug_log.md and JSONL stream."""
    args = ["--config", cfg] if cfg else []
    with command_session("cli.version", args, [cfg] if cfg else None):
        gi = git_info()
        ch = config_hash([cfg]) if cfg else "none"
        info = {
            "cli_version": CLI_VERSION,
            "git": gi,
            "config_hash": ch,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        typer.echo(json.dumps(info, indent=2))


@app.command("test")
def test(
    mode: str = typer.Option("fast", "--mode", help="fast|deep"),
):
    """Run selftest (pipeline/CLI integrity, presence, env snapshot)."""
    rc = selftest_mod.cli(mode=mode)
    raise typer.Exit(rc)


@app.command("analyze-log")
def analyze_log(
    path: str = typer.Option(str(LOG_DIR / "v50_debug_log.md"), "--path", help="Path to MD log"),
    out_md: str = typer.Option(str(REPORTS_DIR / "log_table.md"), "--out-md"),
    out_csv: str = typer.Option(str(REPORTS_DIR / "log_table.csv"), "--out-csv"),
    clean: bool = typer.Option(False, "--clean/--no-clean", help="Deduplicate repeated entries"),
):
    """Parse v50_debug_log.md into a Markdown/CSV table. Optionally deduplicate."""
    with command_session("cli.analyze-log", ["--path", path, "--out-md", out_md, "--out-csv", out_csv, "--clean", str(clean)]):
        p = Path(path)
        if not p.exists():
            typer.echo("Log not found.")
            raise typer.Exit(2)
        text = p.read_text(encoding="utf-8").splitlines()
        entries = []
        cur: dict[str, str] = {}
        for line in text:
            if line.startswith("### ") and " — " in line and "status:" in line:
                if cur:
                    entries.append(cur)
                    cur = {}
                ts = line.split("### ")[1].split(" — ")[0].strip()
                status_match = re.search(r"status:\s+(\S+)", line)
                cur = {"ts": ts, "status": status_match.group(1) if status_match else "?"}
            elif line.strip().startswith("- "):
                k, _, v = line.strip()[2:].partition(":")
                cur[k.strip()] = v.strip().strip("`")
        if cur:
            entries.append(cur)
        if clean:
            seen = set()
            deduped = []
            for e in entries:
                key = (e.get("CLI Version", ""), e.get("Git", ""), e.get("Args", ""), e.get("Config Hash", ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(e)
            entries = deduped
        rows = [["ts", "status", "CLI Version", "Git", "Config Hash", "Args", "Duration"]]
        for e in entries:
            rows.append([
                e.get("ts", ""),
                e.get("status", ""),
                e.get("CLI Version", ""),
                e.get("Git", ""),
                e.get("Config Hash", ""),
                e.get("Args", ""),
                e.get("Duration", ""),
            ])
        md = "# CLI Log Table\n\n" + md_table(rows) + "\n"
        Path(out_md).write_text(md, encoding="utf-8")
        import csv

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0])
            for r in rows[1:]:
                w.writerow(r)
        typer.echo(f"Wrote {out_md}\nWrote {out_csv}")


@app.command("corel-train")
def corel_train(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Hydra config"),
):
    """Train COREL GNN calibrator if available in repo."""
    from .common import find_module_or_script, call_python_module, call_python_file

    with command_session("cli.corel-train", ["--config", str(config or "")], [config] if config else None):
        module = "spectramind.corel.train_corel"
        candidates = [SRC_DIR / "spectramind" / "corel" / "train_corel.py"]
        k, s = find_module_or_script(module, candidates)
        args = ["--config", config] if config else []
        if k == "module":
            rc = call_python_module(module, args)
        elif k == "script" and s:
            rc = call_python_file(s, args)
        else:
            typer.echo("train_corel not found.")
            raise typer.Exit(2)
        raise typer.Exit(rc)


@app.command("check-cli-map")
def check_cli_map():
    """Dump/validate command-to-file mapping via existing utility if present."""
    from .common import find_module_or_script, call_python_module, call_python_file

    with command_session("cli.check-cli-map", []):
        module = "spectramind.cli_explain_util"
        candidates = [SRC_DIR / "spectramind" / "cli_explain_util.py"]
        k, s = find_module_or_script(module, candidates)
        if k == "module":
            rc = call_python_module(module, [])
        elif k == "script" and s:
            rc = call_python_file(s, [])
        else:
            typer.echo("cli_explain_util not found.")
            raise typer.Exit(2)
        raise typer.Exit(rc)


def main():
    app()


if __name__ == "__main__":
    main()
