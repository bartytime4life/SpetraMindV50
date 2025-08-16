from __future__ import annotations

"""
SpectraMind V50 — Submission orchestration CLI with runtime wiring.

Implements a high-level 'make-submission' command that runs predict → validate → bundle,
honoring the user's chosen runtime via Hydra (runtime=<name>).
"""

from typing import Optional, List

import typer

from spectramind.conf_helpers.runtime import (
    determine_runtime,
    set_runtime_env,
    compose_with_runtime,
    log_runtime_choice,
    pretty_print_cfg,
)


app = typer.Typer(help="SpectraMind V50 — Submit CLI")


@app.command("make-submission")
def make_submission(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Execution environment (Hydra group)"
    ),
    config_overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Additional Hydra overrides for submission flow",
        rich_help_panel="Hydra",
    ),
    show_cfg: bool = typer.Option(False, "--show-cfg", help="Print resolved Hydra config and exit"),
):
    """
    Build a leaderboard-ready submission bundle. This command typically:
    1) Runs inference on the test set
    2) Validates the μ/σ outputs and coverage
    3) Packages artifacts (CSV/ZIP/HTML) for upload

    All steps compose a shared Hydra config that includes `runtime=<name>`.
    """
    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    cfg = compose_with_runtime(resolved_runtime, overrides=config_overrides or [])
    log_runtime_choice(resolved_runtime, cfg, extra_note="submit.make-submission")

    if show_cfg:
        pretty_print_cfg(cfg)
        raise typer.Exit(0)

    try:
        from spectramind.cli_submit_impl import make_submission_from_config

        exit_code = make_submission_from_config(cfg)  # type: ignore[call-arg]
    except ModuleNotFoundError:
        typer.echo(
            "[ERROR] Module 'spectramind.cli_submit_impl' not found. "
            "Please include the submission implementation with a 'make_submission_from_config(cfg)' entrypoint."
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Submission workflow failed: {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code if isinstance(exit_code, int) else 0)

