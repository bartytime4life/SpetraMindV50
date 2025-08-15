from pathlib import Path
from typing import List, Optional

import typer

from .common import (
    SRC_DIR,
    logger,
    ensure_tools,
    command_session,
    find_module_or_script,
    call_python_module,
    call_python_file,
)

app = typer.Typer(no_args_is_help=True, help="SpectraMind V50 — End-to-end submission orchestration")


@app.command("make-submission")
def make_submission(
    model_path: str = typer.Option(..., "--model", help="Trained checkpoint"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Hydra config"),
    outdir: str = typer.Option("submission_bundle", "--out", help="Submission output dir"),
    run_selftest: bool = typer.Option(True, "--selftest/--no-selftest"),
):
    """Pipeline:
    1) selftest
    2) predict
    3) calibrate (COREL default)
    4) validate
    5) bundle zip"""
    ensure_tools()
    with command_session("submit.make-submission", ["--model", model_path, "--out", outdir]):
        if run_selftest:
            from .selftest import cli as selftest_cli
            rc = selftest_cli("fast")
            if rc != 0:
                raise typer.Exit(rc)
        module_pred = "spectramind.predict_v50"
        cand_pred = [SRC_DIR / "spectramind" / "predict_v50.py"]
        pred_args: List[str] = ["--model", model_path, "--outdir", "predictions"]
        if config:
            pred_args += ["--config", config]
        k, s = find_module_or_script(module_pred, cand_pred)
        if k == "module":
            rc2 = call_python_module(module_pred, pred_args)
        elif k == "script" and s:
            rc2 = call_python_file(s, pred_args)
        else:
            logger.error("Missing predict_v50.")
            raise typer.Exit(2)
        if rc2 != 0:
            raise typer.Exit(rc2)
        module_cal = "spectramind.uncertainty.calibrate"
        cand_cal = [SRC_DIR / "spectramind" / "uncertainty" / "calibrate.py"]
        k3, s3 = find_module_or_script(module_cal, cand_cal)
        if k3 == "module":
            rc3 = call_python_module(module_cal, ["--preds", "predictions", "--outdir", "calibrated", "--method", "corel"])
        elif k3 == "script" and s3:
            rc3 = call_python_file(s3, ["--preds", "predictions", "--outdir", "calibrated", "--method", "corel"])
        else:
            logger.warning("Calibrator not found; skipping calibration step.")
            rc3 = 0
        if rc3 != 0:
            raise typer.Exit(rc3)
        module_val = "spectramind.validate_submission"
        cand_val = [SRC_DIR / "spectramind" / "validate_submission.py"]
        k4, s4 = find_module_or_script(module_val, cand_val)
        val_args = ["--bundle", outdir]
        if k4 == "module":
            rc4 = call_python_module(module_val, val_args)
        elif k4 == "script" and s4:
            rc4 = call_python_file(s4, val_args)
        else:
            logger.warning("Validator not found; relying on bundler-level checks.")
            rc4 = 0
        if rc4 != 0:
            raise typer.Exit(rc4)
        from .cli_bundle import make as bundle_make
        rc5 = bundle_make(
            preds_dir="calibrated" if k3 != "missing" else "predictions",
            bundle_dir=outdir,
            zip_name="submission.zip",
        )
        raise typer.Exit(rc5 if isinstance(rc5, int) else 0)
