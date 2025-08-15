import logging
from pathlib import Path

import typer
from omegaconf import OmegaConf

from .logging_utils import log_event, setup_logging
from .pipeline import CalibrationPipeline

app = typer.Typer(add_completion=False, help="SpectraMind V50 Calibration CLI")


@app.command("run")
def run(
    config_path: str = typer.Argument(..., help="Path to config_v50.yaml"),
    raw_data: str = typer.Argument(
        "data/raw", help="Directory with raw FGS1/AIRS frames"
    ),
    output: str = typer.Argument("outputs/calibrated", help="Output directory"),
) -> None:
    setup_logging(run_name="calibration")
    logging.getLogger(__name__).info(
        "CLI invoked",
        extra={"config_path": config_path, "raw_data": raw_data, "output": output},
    )
    log_event(
        "cli.start",
        {"config_path": config_path, "raw_data": raw_data, "output": output},
    )

    cfg = OmegaConf.load(config_path)
    pipe_cfg = cfg.get("calibration_pipeline", {})
    pipe = CalibrationPipeline(pipe_cfg)
    paths = cfg.get("paths", {})
    raw_dir = raw_data if raw_data else paths.get("raw_data", "data/raw")
    out_dir = output if output else paths.get("calibrated_output", "outputs/calibrated")
    out_dir = str(Path(out_dir))

    pipe.run(raw_dir, out_dir)
    log_event("cli.complete", {"output": out_dir})
    logging.getLogger(__name__).info("CLI complete", extra={"output": out_dir})


if __name__ == "__main__":  # pragma: no cover
    app()
