#!/usr/bin/env python
import json
import typer
import torch
from typing import Optional

from src.spectramind.diagnostics import selftest_diagnostics
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

from src.spectramind.models import create_fusion

app = typer.Typer(help="SpectraMind V50 unified CLI")

@app.command("fusion-smoke")
def fusion_smoke(
    config: Optional[str] = typer.Option(None, help="Path to fusion YAML (variant)"),
    base:   Optional[str] = typer.Option(None, help="Path to base fusion YAML"),
    ftype:  Optional[str] = typer.Option("concat+mlp", help="Override type if YAMLs not provided"),
    dim:    int = typer.Option(64, help="Model fused dim / also used as d_fgs1/d_airs if no YAMLs"),
):
    if OmegaConf and config and base:
        base_cfg = OmegaConf.load(base)
        var_cfg  = OmegaConf.load(config)
        merged = OmegaConf.merge({"model": {"fusion": {}}}, {"model": {"fusion": dict(base_cfg)}}, {"model": {"fusion": dict(var_cfg)}})
        cfg = OmegaConf.to_container(merged, resolve=True)
    else:
        cfg = {
            "model": {
                "fusion": {
                    "type": ftype,
                    "dim": dim,
                    "dropout": 0.05,
                    "norm": "layernorm",
                    "export": {"taps": True, "attn_weights": True, "gate_values": True},
                    "shapes": {"d_fgs1": dim, "d_airs": dim, "strict_check": True},
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
def diagnose(outdir: str = typer.Option("diagnostics", help="Output directory")):
    """Run SpectraMind diagnostics self-test."""
    selftest_diagnostics.run_selftest(outdir)

if __name__ == "__main__":
    app()
