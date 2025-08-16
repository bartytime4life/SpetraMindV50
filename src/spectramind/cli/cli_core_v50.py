from pathlib import Path
from typing import List, Optional

import typer

from .common import (
    PROJECT_ROOT,
    SRC_DIR,
    REPORTS_DIR,
    logger,
    ensure_tools,
    command_session,
    find_module_or_script,
    call_python_module,
    call_python_file,
    hydra_kv_to_cli,
)
from .cli_guardrails import dry_run_guard

app = typer.Typer(no_args_is_help=True, help="SpectraMind V50 — Core commands: train, predict, calibrate, validate, explain.")


def _cfg_list(config: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if config:
        p = Path(config)
        if p.exists():
            paths.append(p)
    return paths


@app.command()
@dry_run_guard
def train(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Hydra YAML config (config_v50.yaml)"),
    overrides: List[str] = typer.Argument(None, help="Hydra overrides key=value"),
    dry_run: bool = False,
):
    """Train SpectraMind V50 using Hydra config. Supports Hydra overrides: e.g. train epochs=50 optim.lr=3e-4"""
    ensure_tools()
    runtime = ctx.obj.get("runtime", "default")
    args = [f"runtime={runtime}", *hydra_kv_to_cli(overrides or [])]
    with command_session("core.train", ["--config", str(config or ""), *args], _cfg_list(config)):
        module = "spectramind.train_v50"
        candidates = [SRC_DIR / "spectramind" / "train_v50.py"]
        kind, script = find_module_or_script(module, candidates)
        ret = 0
        if kind == "module":
            ret = call_python_module(module, (["--config", str(config), *args]) if config else args)
        elif kind == "script" and script:
            ret = call_python_file(script, (["--config", str(config), *args]) if config else args)
        else:
            logger.error("Missing training entrypoint (spectramind.train_v50).")
            raise typer.Exit(2)
        raise typer.Exit(ret)


@app.command()
@dry_run_guard
def predict(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model", help="Path to trained model weights/checkpoint"),
    outdir: str = typer.Option("predictions", "--outdir", help="Directory for predictions"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Optional Hydra config for inference"),
    overrides: List[str] = typer.Argument(None, help="Hydra overrides key=value"),
    dry_run: bool = False,
):
    """Run prediction to produce μ and σ spectra and artifacts for submission packaging."""
    ensure_tools()
    runtime = ctx.obj.get("runtime", "default")
    args = ["--model", model_path, "--outdir", outdir]
    if config:
        args += ["--config", config]
    args += [f"runtime={runtime}"]
    args += hydra_kv_to_cli(overrides or [])
    with command_session("core.predict", args, _cfg_list(config)):
        module = "spectramind.predict_v50"
        candidates = [SRC_DIR / "spectramind" / "predict_v50.py"]
        kind, script = find_module_or_script(module, candidates)
        if kind == "module":
            ret = call_python_module(module, args)
        elif kind == "script" and script:
            ret = call_python_file(script, args)
        else:
            logger.error("Missing prediction entrypoint (spectramind.predict_v50).")
            raise typer.Exit(2)
        raise typer.Exit(ret)


@app.command()
@dry_run_guard
def calibrate(
    ctx: typer.Context,
    preds_dir: str = typer.Option("predictions", "--preds", help="Directory with raw μ/σ"),
    outdir: str = typer.Option("calibrated", "--outdir", help="Directory for calibrated outputs"),
    method: str = typer.Option("corel", "--method", help="Calibration method: corel|temp|hybrid"),
    dry_run: bool = False,
):
    """Calibrate predictive uncertainty (σ) using COREL / temperature scaling / hybrid methods."""
    ensure_tools()
    runtime = ctx.obj.get("runtime", "default")
    args = ["--preds", preds_dir, "--outdir", outdir, "--method", method, f"runtime={runtime}"]
    with command_session("core.calibrate", args):
        module = "spectramind.uncertainty.calibrate"
        candidates = [SRC_DIR / "spectramind" / "uncertainty" / "calibrate.py"]
        kind, script = find_module_or_script(module, candidates)
        if kind == "module":
            ret = call_python_module(module, args)
        elif kind == "script" and script:
            ret = call_python_file(script, args)
        else:
            logger.error("Missing calibrator (spectramind.uncertainty.calibrate).")
            raise typer.Exit(2)
        raise typer.Exit(ret)


@app.command()
@dry_run_guard
def validate(
    ctx: typer.Context,
    bundle_dir: str = typer.Option("submission_bundle", "--bundle", help="Submission directory to validate"),
    dry_run: bool = False,
):
    """Validate submission bundle: file presence, GLL score evaluator, zips, and schema checks."""
    ensure_tools()
    runtime = ctx.obj.get("runtime", "default")
    args = ["--bundle", bundle_dir, f"runtime={runtime}"]
    with command_session("core.validate", args):
        module = "spectramind.validate_submission"
        candidates = [SRC_DIR / "spectramind" / "validate_submission.py"]
        kind, script = find_module_or_script(module, candidates)
        if kind == "module":
            ret = call_python_module(module, args)
        elif kind == "script" and script:
            ret = call_python_file(script, args)
        else:
            module2 = "spectramind.generate_diagnostic_summary"
            cand2 = [SRC_DIR / "spectramind" / "generate_diagnostic_summary.py"]
            k2, s2 = find_module_or_script(module2, cand2)
            if k2 == "module":
                ret = call_python_module(module2, ["--validate-bundle", bundle_dir])
            elif k2 == "script" and s2:
                ret = call_python_file(s2, ["--validate-bundle", bundle_dir])
            else:
                logger.error("No validator found.")
                raise typer.Exit(2)
        raise typer.Exit(ret)


@app.command("explain")
@dry_run_guard
def explain_shap(
    ctx: typer.Context,
    preds_dir: str = typer.Option("predictions", "--preds", help="Predictions directory for SHAP/metadata explain"),
    outdir: str = typer.Option("explain", "--outdir", help="Output directory"),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard", help="Generate interactive HTML dashboard"),
    dry_run: bool = False,
):
    """Run SHAP + metadata + symbolic overlays and optionally generate a dashboard."""
    ensure_tools()
    runtime = ctx.obj.get("runtime", "default")
    args = ["--preds", preds_dir, "--outdir", outdir, f"runtime={runtime}"]
    if dashboard:
        args += ["--dashboard"]
    with command_session("core.explain", args):
        module = "spectramind.explain_shap_metadata_v50"
        candidates = [SRC_DIR / "spectramind" / "explain_shap_metadata_v50.py"]
        kind, script = find_module_or_script(module, candidates)
        if kind == "module":
            ret = call_python_module(module, args)
        elif kind == "script" and script:
            ret = call_python_file(script, args)
        else:
            logger.error("Missing explainer (explain_shap_metadata_v50).")
            raise typer.Exit(2)
        raise typer.Exit(ret)
