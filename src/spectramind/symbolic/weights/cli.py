"""Typer CLI for the weights subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .auto_weight_optimizer import OptimizationConfig, optimize_symbolic_weights
from .loader import compose_weights, list_available_weight_sets
from .logging_utils import get_logger
from .validator import validate_weight_config

app = typer.Typer(no_args_is_help=True, add_completion=True)
LOGGER = get_logger(__name__)


@app.command("list")
def list_weights() -> None:
    """List available YAML weight files shipped with this package."""
    for name in list_available_weight_sets():
        typer.echo(name)


@app.command("compose")
def compose(
    profile_name: Optional[str] = typer.Option(
        None, "--profile-name", help="Name inside composite_profile_weights.yaml"
    ),
    profile_file: Optional[str] = typer.Option(
        None, "--profile-file", help="profiles/*.yaml or absolute path"
    ),
    base_files: Optional[List[str]] = typer.Option(
        None,
        "--base",
        help="Explicit base/penalty YAML list (defaults used if omitted)",
    ),
    directory: Optional[Path] = typer.Option(
        None, "--dir", help="Directory containing YAML files (default: package dir)"
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out", help="Save composed weights to JSON file"
    ),
) -> None:
    """Compose final weights (base + penalties + molecule + profile) and print as JSON."""
    weights = compose_weights(
        base_files=base_files,
        directory=directory,
        profile_file=profile_file,
        profile_name=profile_name,
    )
    validate_weight_config(weights)
    out = json.dumps(weights, indent=2, sort_keys=True)
    typer.echo(out)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(out, encoding="utf-8")
        typer.echo(f"Wrote {json_out}")


@app.command("optimize")
def optimize(
    weights_json: Path = typer.Argument(
        ..., help="Input weights JSON (from compose or hand-authored)"
    ),
    metrics_json: Path = typer.Argument(
        ..., help="Performance metrics JSON {rule: metric}"
    ),
    target: float = typer.Option(1.0, help="Target metric value"),
    step: float = typer.Option(0.10, help="Base proportional step"),
    min_delta: float = typer.Option(-0.25, help="Min relative change"),
    max_delta: float = typer.Option(0.25, help="Max relative change"),
    clip_min: float = typer.Option(0.0, help="Absolute minimum"),
    clip_max: float = typer.Option(10.0, help="Absolute maximum"),
    dry_run: bool = typer.Option(False, help="Preview adjustments without applying"),
    out_json: Optional[Path] = typer.Option(
        None, "--out-json", help="Save optimized weights JSON here"
    ),
) -> None:
    """Optimize weights given metrics; prints updated JSON (or preview if --dry-run)."""
    weights = json.loads(weights_json.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))

    cfg = OptimizationConfig(
        target=target,
        step=step,
        min_delta=min_delta,
        max_delta=max_delta,
        absolute_clip_min=clip_min,
        absolute_clip_max=clip_max,
        dry_run=dry_run,
    )
    updated = optimize_symbolic_weights(weights, metrics, cfg)
    validate_weight_config(updated)
    out = json.dumps(updated, indent=2, sort_keys=True)
    typer.echo(out)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(out, encoding="utf-8")
        typer.echo(f"Wrote {out_json}")


if __name__ == "__main__":  # pragma: no cover
    app()
