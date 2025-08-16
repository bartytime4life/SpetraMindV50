#!/usr/bin/env python
import json
from typing import List, Optional

import torch
import typer

from src.spectramind.diagnostics import selftest_diagnostics

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional dependency
    OmegaConf = None

from spectramind.conf_helpers import (
    apply_overrides,
    cli_override_parser,
    inject_symbolic_constraints,
    load_config,
    log_environment,
    validate_config,
)
from src.spectramind.models import create_fusion

app = typer.Typer(help="SpectraMind V50 unified CLI")

config_app = typer.Typer(
    help="Configuration utilities (Hydra, overrides, validation, env)"
)
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate(config: str, schema: str) -> None:
    """Validate a YAML config against a JSON schema."""
    cfg = load_config(config)
    ok = validate_config(cfg, schema)
    if ok:
        typer.echo(f"\u2705 Config {config} is valid against schema {schema}")


@config_app.command("apply-override")
def config_apply_override(config: str, overrides: List[str]) -> None:
    """Apply CLI-style overrides to a config and print result."""
    cfg = load_config(config)
    parsed = cli_override_parser(overrides)
    cfg = apply_overrides(cfg, parsed)
    typer.echo(cfg.pretty())


@config_app.command("capture-env")
def config_capture_env(out: str = "env_capture.json") -> None:
    """Capture environment metadata and save to file."""
    env = log_environment(out)
    typer.echo(f"\u2705 Environment captured in {out}")
    typer.echo(env)


@config_app.command("inject-symbolic")
def config_inject_symbolic(config: str) -> None:
    """Inject default symbolic constraint weights into a config."""
    cfg = load_config(config)
    cfg = inject_symbolic_constraints(cfg)
    typer.echo(cfg.pretty())


@app.command("fusion-smoke")
def fusion_smoke(
    config: Optional[str] = typer.Option(None, help="Path to fusion YAML (variant)"),
    base: Optional[str] = typer.Option(None, help="Path to base fusion YAML"),
    ftype: Optional[str] = typer.Option(
        "concat+mlp", help="Override type if YAMLs not provided"
    ),
    dim: int = typer.Option(
        64,
        help="Model fused dim / also used as d_fgs1/d_airs if no YAMLs",
    ),
) -> None:
    if OmegaConf and config and base:
        base_cfg = OmegaConf.load(base)
        var_cfg = OmegaConf.load(config)
        merged = OmegaConf.merge(
            {"model": {"fusion": {}}},
            {"model": {"fusion": dict(base_cfg)}},
            {"model": {"fusion": dict(var_cfg)}},
        )
        cfg = OmegaConf.to_container(merged, resolve=True)
    else:
        cfg = {
            "model": {
                "fusion": {
                    "type": ftype,
                    "dim": dim,
                    "dropout": 0.05,
                    "norm": "layernorm",
                    "export": {
                        "taps": True,
                        "attn_weights": True,
                        "gate_values": True,
                    },
                    "shapes": {
                        "d_fgs1": dim,
                        "d_airs": dim,
                        "strict_check": True,
                    },
                }
            }
        }
    fuser = create_fusion(cfg)
    B = 4
    h_fgs1 = torch.randn(B, fuser.d_fgs1)
    h_airs = torch.randn(B, fuser.d_airs)
    fused, extras = fuser(h_fgs1, h_airs, molecule=torch.ones(B), seam=torch.zeros(B))
    out = {"fused_shape": list(fused.shape), "extras_keys": list(extras.keys())}
    typer.echo(json.dumps(out))


@app.command("diagnose")
def diagnose(
    outdir: str = typer.Option("diagnostics", help="Output directory"),
) -> None:
    """Run SpectraMind diagnostics self-test."""
    selftest_diagnostics.run_selftest(outdir)


if __name__ == "__main__":
    app()
