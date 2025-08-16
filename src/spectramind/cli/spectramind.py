"""
SpectraMind V50 - Unified CLI (Root)

* Registers core commands and integrates Telemetry automatically.
* Each command is wrapped with @with_telemetry to log start/end/duration/config hash.

This CLI intentionally keeps commands minimal here but can dispatch into the
project's richer command modules (train/predict/diagnose/submit/etc.).
"""

import json
import os
import time

import typer

from spectramind.cli.cli_telemetry import with_telemetry
from spectramind.telemetry import TelemetryManager

app = typer.Typer(help="SpectraMind V50 - Unified CLI with Telemetry")

SPECTRAMIND_CLI_VERSION = "v50.0.0-telemetry"
os.environ.setdefault("SPECTRAMIND_CLI_VERSION", SPECTRAMIND_CLI_VERSION)


@app.callback()
def main_callback() -> None:
    """
    Root callback — sets ENV defaults for telemetry/logging so subcommands inherit.
    """
    os.environ.setdefault("SPECTRAMIND_LOG_DIR", "logs/telemetry")
    os.environ.setdefault("SPECTRAMIND_ENABLE_JSONL", "1")


@app.command("version")
@with_telemetry("version")
def version() -> None:
    """
    Print CLI version and any available config hash.
    """
    info = {
        "cli_version": os.environ.get("SPECTRAMIND_CLI_VERSION"),
        "config_hash": os.environ.get("SPECTRAMIND_CONFIG_HASH"),
        "timestamp": time.time(),
    }
    typer.echo(json.dumps(info, indent=2))


@app.command("test")
@with_telemetry("test")
def run_selftest(
    fast: bool = typer.Option(False, "--fast", help="Run fast checks only"),
    deep: bool = typer.Option(False, "--deep", help="Run deep checks"),
) -> None:
    """
    Placeholder self-test hook. In your repo this should delegate to `selftest.py`.
    Here we simply emit a telemetry diagnostic sample.
    """
    tm = TelemetryManager()
    tm.log_metric("selftest_fast_flag", 1.0 if fast else 0.0, step=0)
    tm.log_metric("selftest_deep_flag", 1.0 if deep else 0.0, step=0)

    tm.attach_diagnostics(lambda: {"telemetry_ok": True, "fast": fast, "deep": deep})
    diag = tm.run_diagnostics()
    typer.echo(json.dumps({"selftest": "ok", "diagnostics": diag}, indent=2))


@app.command("diagnose-log")
@with_telemetry("diagnose-log")
def diagnose_log(
    log_dir: str = typer.Option(
        "logs/telemetry", "--log-dir", help="Telemetry logs directory"
    ),
) -> None:
    """
    Summarize recent telemetry JSONL session files (very lightweight).
    """
    import glob

    paths = sorted(glob.glob(os.path.join(log_dir, "*.jsonl")))
    out = {"files": len(paths), "events": 0}
    for p in paths[-5:]:
        with open(p, "r", encoding="utf-8") as f:
            out["events"] += sum(1 for _ in f)
    typer.echo(json.dumps(out, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
