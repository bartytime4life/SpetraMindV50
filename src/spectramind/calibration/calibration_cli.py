"""Typer-based command line interface for the calibration package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from .calibration_checker import evaluate_sigma_calibration, save_summary_json
from .calibration_config import CalibrationConfig
from .corel_calibration import conformal_calibrate_sigma
from .git_env_capture import reproducibility_snapshot
from .logging_utils import append_debug_md, setup_logging
from .pipeline import run_calibration_batch, run_calibration_one
from .temperature_scaling import apply_temperature_scaling, fit_temperature_scaling

app = typer.Typer(no_args_is_help=True, add_completion=False, help="SpectraMind V50 Calibration CLI")


@app.command("run")
def cmd_run(
    cube_path: str = typer.Argument(..., help="Path to .npy cube (T,H,W)"),
    instrument: str = typer.Option("AIRS", help="Instrument name"),
    exposure_s: float = typer.Option(1.0, help="Exposure duration in seconds"),
    config_yaml: Optional[str] = typer.Option(None, help="Hydra/OmegaConf YAML for calibration config"),
    output_dir: Optional[str] = typer.Option(None, help="Override output dir"),
) -> None:
    cfg = CalibrationConfig.from_yaml(config_yaml) if config_yaml else CalibrationConfig()
    if output_dir:
        cfg.io.output_dir = output_dir
    res = run_calibration_one(
        cube_path=cube_path, instrument=instrument, cfg=cfg, exposure_s=exposure_s
    )
    typer.echo(json.dumps({"instrument": instrument, "cr_replaced": res.get("cr_replaced", 0)}, indent=2))


@app.command("batch")
def cmd_batch(
    input_dir: str = typer.Argument(..., help="Directory containing .npy cubes"),
    exposure_s: float = typer.Option(1.0, help="Exposure duration in seconds"),
    config_yaml: Optional[str] = typer.Option(None, help="Hydra/OmegaConf YAML for calibration config"),
    output_dir: Optional[str] = typer.Option(None, help="Override output dir"),
) -> None:
    cfg = CalibrationConfig.from_yaml(config_yaml) if config_yaml else CalibrationConfig()
    if output_dir:
        cfg.io.output_dir = output_dir
    manifest = run_calibration_batch(input_dir=input_dir, cfg=cfg, exposure_s=exposure_s)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("sigma-calibrate")
def cmd_sigma_calibrate(
    mu_pred_path: str = typer.Argument(..., help="Path to μ predictions .npy"),
    mu_true_path: str = typer.Argument(..., help="Path to μ ground truth .npy"),
    sigma_pred_path: str = typer.Argument(..., help="Path to σ predictions .npy"),
    alpha: float = typer.Option(0.1, help="Conformal risk level"),
    out_sigma_path: str = typer.Option("artifacts/calibration/sigma_calibrated.npy", help="Output path"),
    temperature: bool = typer.Option(True, help="Fit global temperature scaling first"),
) -> None:
    logger, evt = setup_logging()
    mu_pred = np.load(mu_pred_path)
    mu_true = np.load(mu_true_path)
    sigma_pred = np.load(sigma_pred_path)

    if temperature:
        T = fit_temperature_scaling(mu_pred - mu_true, sigma_pred)
        sigma_pred = apply_temperature_scaling(sigma_pred, T)
        evt.log({"event": "temperature_scaling", "T": float(T)})
        logger.info(f"Temperature scaling T={T:.4f}")

    sigma_cal, meta = conformal_calibrate_sigma(mu_pred, mu_true, sigma_pred, alpha=alpha)
    Path(out_sigma_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_sigma_path, sigma_cal)

    summary = evaluate_sigma_calibration(mu_pred, mu_true, sigma_cal)
    out_json = str(Path(out_sigma_path).with_suffix(".json"))
    save_summary_json(summary | {"meta": meta}, out_json)
    typer.echo(json.dumps({"out_sigma": out_sigma_path, "summary": summary, "meta": meta}, indent=2))


@app.command("validate")
def cmd_validate(
    mu_pred_path: str = typer.Argument(...),
    mu_true_path: str = typer.Argument(...),
    sigma_path: str = typer.Argument(...),
    out_json: str = typer.Option("artifacts/calibration/calibration_summary.json", help="Summary JSON"),
) -> None:
    mu_pred = np.load(mu_pred_path)
    mu_true = np.load(mu_true_path)
    sigma = np.load(sigma_path)
    summary = evaluate_sigma_calibration(mu_pred, mu_true, sigma)
    save_summary_json(summary, out_json)
    typer.echo(json.dumps(summary, indent=2))


@app.command("selftest")
def cmd_selftest() -> None:
    logger, evt = setup_logging()
    append_debug_md("Calibration Self-Test", reproducibility_snapshot())
    T, H, W = 64, 32, 32
    rng = np.random.default_rng(1337)
    base = rng.normal(1000, 5, size=(T, H, W)).astype(np.float32)
    y, x = np.mgrid[0:H, 0:W]
    psf = np.exp(-((x - W / 2) ** 2 + (y - H / 2) ** 2) / (2 * 3.0 ** 2))
    psf /= psf.sum()
    for t in range(T):
        base[t] += 500 * psf
    flux_true = np.ones(T)
    flux_true[24:40] -= 0.01
    for t in range(T):
        base[t] *= flux_true[t]
    cfg = CalibrationConfig()
    out_dir = Path(cfg.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cube_path = out_dir / "SELF_AIRS_cube.npy"
    np.save(cube_path, base)
    res = run_calibration_one(str(cube_path), instrument="AIRS", cfg=cfg, exposure_s=1.0)
    typer.echo(json.dumps({"ok": True, "cr_replaced": res.get("cr_replaced", 0)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
