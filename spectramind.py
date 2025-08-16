from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import OmegaConf

from spectramind.conf_helpers import (
    apply_overrides,
    cli_override_parser,
    inject_symbolic_constraints,
    load_config,
    log_environment,
    run_config_audit,
    validate_config,
)
from spectramind.diagnostics.config_audit_pipeline import run_config_audit_and_save
from spectramind.reporting.generate_html_report import generate_dashboard_html

app = typer.Typer(help="SpectraMind V50 Unified CLI")


# -----------------------
# Sub-CLI: config
# -----------------------
config_app = typer.Typer(
    help="Configuration utilities (Hydra, overrides, validation, env, audit)"
)
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate(config: str, schema: str):
    """
    Validate a YAML/Hydra config against a JSON/YAML schema.
    """
    cfg = load_config(config)
    validate_config(cfg, schema)
    typer.echo(f"✅ Config '{config}' is valid against schema '{schema}'")


@config_app.command("apply-override")
def config_apply_override(config: str, overrides: List[str]):
    """
    Apply CLI-style overrides to a config and print result (YAML).
    """
    cfg = load_config(config)
    parsed = cli_override_parser(overrides)
    cfg = apply_overrides(cfg, parsed)
    out = OmegaConf.to_yaml(cfg, resolve=True)
    typer.echo(out)


@config_app.command("inject-symbolic")
def config_inject_symbolic(config: str):
    """
    Inject default symbolic constraint weights into a config and print (YAML).
    """
    cfg = load_config(config)
    cfg = inject_symbolic_constraints(cfg)
    out = OmegaConf.to_yaml(cfg, resolve=True)
    typer.echo(out)


@config_app.command("capture-env")
def config_capture_env(
    out: str = "artifacts/diagnostics/env_capture.json", detailed: bool = True
):
    """
    Capture environment metadata and save to file (JSON). Detailed includes GPU + pip freeze.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    env = log_environment(out, detailed=detailed)
    typer.echo(f"✅ Environment captured: {out}")
    typer.echo(env)


@config_app.command("audit")
def config_audit(
    config: str,
    schema: Optional[str] = typer.Option(None),
    overrides: List[str] = typer.Option(default=[]),
    out: str = typer.Option("artifacts/diagnostics/config_audit.json"),
):
    """
    Run a configuration audit (load+overrides+schema validation+symbolic defaults+env capture)
    and save JSON for dashboard consumption.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run_config_audit(config, schema, overrides, out_json=out)
    typer.echo(f"✅ Config audit saved: {out}")


# -----------------------
# Sub-CLI: diagnose
# -----------------------
diagnose_app = typer.Typer(help="Diagnostics utilities (dashboard, audits)")
app.add_typer(diagnose_app, name="diagnose")


@diagnose_app.command("dashboard")
def diagnose_dashboard(
    config: str = typer.Option(..., help="YAML path or Hydra name"),
    schema: Optional[str] = typer.Option(None, help="Schema path (JSON or YAML)"),
    overrides: List[str] = typer.Option(default=[], help="Hydra-style overrides"),
    out_html: str = typer.Option(
        "artifacts/diagnostics/dashboard.html", help="Output HTML path"
    ),
    title: str = typer.Option("SpectraMind V50 Diagnostics Dashboard"),
):
    """
    Build the diagnostics dashboard HTML, including the Configuration Audit section.

    Steps:
      1) Run config audit -> writes artifacts/diagnostics/config_audit.json
      2) Generate HTML dashboard embedding the audit
    """
    audit_json = run_config_audit_and_save(
        config_path_or_name=config, schema_path=schema, overrides=overrides
    )
    out = generate_dashboard_html(
        out_html=out_html, title=title, config_audit_json=audit_json
    )
    typer.echo(f"✅ Dashboard generated: {out}")


if __name__ == "__main__":
    app()
