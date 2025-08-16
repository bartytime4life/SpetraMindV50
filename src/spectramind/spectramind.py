from __future__ import annotations

"""
SpectraMind V50 — Root Typer CLI with global --runtime wiring.

This file registers sub-CLIs and ensures that any command run beneath the root
inherits the chosen runtime environment (Hydra group override runtime=<name>).

Sub-CLIs mounted:
    • core   : Training / inference / utilities (cli_core_v50)
    • submit : Submission orchestration (cli_submit)
    • diagnose: Diagnostics, dashboards, explainability (cli_diagnose)
    • test   : Self-test runner (spectramind test) for quick repository integrity checks

Global options:
--runtime [default|local|kaggle|hpc|docker|ci]
Applies a Hydra override runtime=<name> across all subcommands and calls.
"""

import os
import sys
from typing import Optional

import typer

from spectramind.conf_helpers.runtime import (
    determine_runtime,
    set_runtime_env,
    log_runtime_choice,
)

# Import sub-apps (they themselves also accept/resolve --runtime, but mounting them
# under this root enables a single global flag to apply everywhere).
from spectramind.cli import cli_core_v50 as _cli_core_v50
from spectramind.cli import cli_submit as _cli_submit
from spectramind.cli import cli_diagnose as _cli_diagnose

app = typer.Typer(help="SpectraMind V50 — Unified CLI (root)", add_completion=True)


def _version_line() -> str:
    """
    Return a compact version string. We avoid heavy imports here.
    Extend this function to include run hash if you maintain run_hash_summary_v50.json.
    """
    cli_version = "v50"
    return f"SpectraMind CLI {cli_version}"


@app.callback()
def main(
    ctx: typer.Context,
    runtime: Optional[str] = typer.Option(
        None,
        "--runtime",
        "-r",
        help="Select execution environment (Hydra group): default|local|kaggle|hpc|docker|ci",
    ),
    version: bool = typer.Option(False, "--version", help="Show CLI version and exit"),
):
    """
    Root callback runs once and sets SPECTRAMIND_RUNTIME so all nested Typer apps
    and subprocesses inherit it. This achieves a global 'runtime' switch behavior.
    """
    if version:
        typer.echo(_version_line())
        raise typer.Exit(0)

    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    # Lightweight, best-effort log line (will not raise if file permissions are restricted).
    log_runtime_choice(resolved_runtime, cfg=None, extra_note="root-cli")  # cfg is optional here


# Mount sub-CLIs under the root.
app.add_typer(_cli_core_v50.app, name="core", help="Core training/inference CLI")
app.add_typer(_cli_submit.app, name="submit", help="Submission orchestration CLI")
app.add_typer(_cli_diagnose.app, name="diagnose", help="Diagnostics and dashboard CLI")


@app.command("test")
def run_selftest(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Override execution environment for the self-test"
    ),
):
    """
    Execute repository self-checks. Honors the same global runtime selection.
    """
    # Re-resolve runtime if provided here; otherwise use ENV set by root callback.
    from spectramind.conf_helpers.runtime import determine_runtime, set_runtime_env, log_runtime_choice

    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    log_runtime_choice(resolved_runtime, cfg=None, extra_note="selftest")

    # Defer import to keep CLI import time minimal.
    try:
        from spectramind.selftest import main as selftest_main
    except Exception as e:
        typer.echo(f"[ERROR] Failed to import selftest module: {e}", err=True)
        raise typer.Exit(1)

    # Execute the self-test routine (it composes Hydra configs with runtime override internally).
    exit_code = selftest_main()
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()

