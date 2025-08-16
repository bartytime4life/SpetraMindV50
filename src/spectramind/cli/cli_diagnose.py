from __future__ import annotations

"""
SpectraMind V50 — Diagnostics & Dashboard CLI with runtime wiring.

Provides dashboard, symbolic-rank, and other diagnostic commands, all honoring
the selected Hydra runtime group via runtime=<name>.
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


app = typer.Typer(help="SpectraMind V50 — Diagnostics CLI")


@app.command("dashboard")
def dashboard(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Execution environment (Hydra group)"
    ),
    config_overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Additional Hydra overrides for HTML dashboard generation",
        rich_help_panel="Hydra",
    ),
    show_cfg: bool = typer.Option(False, "--show-cfg", help="Print resolved Hydra config and exit"),
):
    """
    Generate the interactive diagnostics dashboard HTML (UMAP/t-SNE/SHAP/symbolic overlays).
    """
    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    cfg = compose_with_runtime(resolved_runtime, overrides=config_overrides or [])
    log_runtime_choice(resolved_runtime, cfg, extra_note="diagnose.dashboard")

    if show_cfg:
        pretty_print_cfg(cfg)
        raise typer.Exit(0)

    try:
        from spectramind.diagnostics.generate_html_report import generate_report_from_config

        exit_code = generate_report_from_config(cfg)  # type: ignore[call-arg]
    except ModuleNotFoundError:
        typer.echo(
            "[ERROR] Module 'spectramind.diagnostics.generate_html_report' not found. "
            "Please ensure your repository includes the diagnostics dashboard implementation."
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Dashboard generation failed: {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code if isinstance(exit_code, int) else 0)


@app.command("symbolic-rank")
def symbolic_rank(
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Execution environment (Hydra group)"
    ),
    config_overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Additional Hydra overrides for symbolic analysis",
        rich_help_panel="Hydra",
    ),
    show_cfg: bool = typer.Option(False, "--show-cfg", help="Print resolved Hydra config and exit"),
):
    """
    Run the symbolic rule violation analysis and export leaderboards / overlays.
    """
    resolved_runtime = determine_runtime(runtime)
    set_runtime_env(resolved_runtime)
    cfg = compose_with_runtime(resolved_runtime, overrides=config_overrides or [])
    log_runtime_choice(resolved_runtime, cfg, extra_note="diagnose.symbolic-rank")

    if show_cfg:
        pretty_print_cfg(cfg)
        raise typer.Exit(0)

    try:
        from spectramind.symbolic.symbolic_violation_predictor import run_symbolic_rank_from_config

        exit_code = run_symbolic_rank_from_config(cfg)  # type: ignore[call-arg]
    except ModuleNotFoundError:
        typer.echo(
            "[ERROR] Module 'spectramind.symbolic.symbolic_violation_predictor' not found. "
            "Ensure your repository includes the symbolic ranking implementation."
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] Symbolic ranking failed: {e}")
        raise typer.Exit(1)

    raise typer.Exit(exit_code if isinstance(exit_code, int) else 0)

