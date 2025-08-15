import argparse
import logging
import os
from typing import Optional

import numpy as np

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, save_json
from ._logging import (
    capture_env_and_git,
    get_logger,
    log_event,
    maybe_mlflow_start_run,
    maybe_wandb_init,
)
from .check_calibration import calibration_metrics
from .gll_error_localizer import gll_contributions
from .spectral_smoothness_map import second_derivative_l2


def run(
    cfg: DiagnosticsConfig,
    config_path: Optional[str] = None,
    run_name: str = "generate-diagnostic-summary",
) -> str:
    """Generate a compact diagnostic summary JSON."""
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "diag-summary-start",
        {"pred_dir": cfg.io.pred_dir},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )

    with maybe_mlflow_start_run(cfg.sync.mlflow, run_name, logger):
        wandb = maybe_wandb_init(
            cfg.sync.wandb,
            cfg.sync.wandb_project,
            run_name,
            config={"config_path": config_path},
            logger=logger,
        )

        mu, sigma, y_true = load_preds_truth(
            cfg.io.pred_dir, cfg.io.y_true, cfg.io.mu_name, cfg.io.sigma_name
        )
        out_dir = cfg.io.output_dir
        ensure_dir(out_dir)

        gll_mean = rmse = mae = calib = None
        if y_true is not None:
            gll = gll_contributions(y_true, mu, sigma)
            gll_mean = float(gll.mean())
            rmse = float(np.sqrt(np.mean((y_true - mu) ** 2)))
            mae = float(np.mean(np.abs(y_true - mu)))
            calib = calibration_metrics(mu, sigma, y_true)

        sm = second_derivative_l2(mu)
        sm_stats = {
            "mean": float(sm.mean()),
            "p90": float(np.percentile(sm, 90)),
            "p99": float(np.percentile(sm, 99)),
        }

        summary = {
            "paths": {
                "pred_dir": cfg.io.pred_dir,
                "output_dir": cfg.io.output_dir,
                "config_path": config_path,
            },
            "metrics": {
                "gll_mean": gll_mean,
                "rmse": rmse,
                "mae": mae,
                "smoothness": sm_stats,
                "calibration": calib,
            },
        }
        out_json = os.path.join(out_dir, "diagnostic_summary.json")
        save_json(summary, out_json)

        try:
            tolog = {k: v for k, v in summary["metrics"].items() if v is not None}
            if "smoothness" in tolog and isinstance(tolog["smoothness"], dict):
                for sk, sv in list(tolog["smoothness"].items()):
                    tolog[f"smoothness_{sk}"] = sv
                del tolog["smoothness"]
            if "calibration" in tolog and isinstance(tolog["calibration"], dict):
                for ck, cv in list(tolog["calibration"].items()):
                    tolog[f"calibration_{ck}"] = cv
                del tolog["calibration"]
            wandb.log(tolog)
        except Exception:  # noqa: BLE001
            pass

        log_event(
            "diag-summary-finish",
            {"json": out_json},
            log_dir=cfg.log.log_dir,
            logger=logger,
        )
        try:
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Diagnostic summary written: %s", out_json)
        return out_json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SpectraMind V50 — Generate Diagnostic Summary"
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
