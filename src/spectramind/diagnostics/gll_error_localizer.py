import argparse
import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, save_csv
from ._logging import (
    capture_env_and_git,
    get_logger,
    log_event,
    maybe_mlflow_start_run,
    maybe_wandb_init,
)


def gaussian_log_likelihood(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Bin-wise Gaussian log-likelihood (GLL). Higher is better."""
    s2 = np.maximum(sigma**2, eps)
    return -0.5 * (np.log(2.0 * np.pi * s2) + ((y - mu) ** 2) / s2)


def gll_contributions(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return gaussian_log_likelihood(y, mu, sigma)


def plot_gll_heatmap_per_bin(
    gll: np.ndarray,
    out_png: str,
    title: str = "Bin-wise Gaussian Log-Likelihood (higher is better)",
    dpi: int = 120,
) -> None:
    """Heatmap: rows=objects, cols=bins."""
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(12, 6), dpi=dpi)
    im = plt.imshow(gll, aspect="auto", interpolation="nearest", cmap="coolwarm")
    plt.colorbar(im, label="GLL")
    plt.xlabel("Bin index")
    plt.ylabel("Object index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run(
    cfg: DiagnosticsConfig,
    config_path: Optional[str] = None,
    run_name: str = "gll-error-localizer",
) -> str:
    """Compute GLL, export per-bin CSV summaries and an overall heatmap."""
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "gll-localizer-start",
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
        if y_true is None:
            raise RuntimeError(
                "y_true is required for GLL error localization but was not provided."
            )

        gll = gll_contributions(y_true, mu, sigma)
        per_bin_mean = gll.mean(axis=0)
        worst_bins_idx = np.argsort(per_bin_mean)

        df = pd.DataFrame(
            {
                "bin": np.arange(gll.shape[1]),
                "mean_gll": per_bin_mean,
                "rank": np.argsort(np.argsort(-per_bin_mean)) + 1,
            }
        )
        out_csv = os.path.join(cfg.io.output_dir, "gll_per_bin_summary.csv")
        save_csv(df, out_csv)

        out_png = os.path.join(cfg.io.output_dir, "gll_heatmap.png")
        plot_gll_heatmap_per_bin(gll, out_png, dpi=120)

        worst10 = worst_bins_idx[:10].tolist()
        log_event(
            "gll-worst-bins",
            {"worst10": worst10},
            log_dir=cfg.log.log_dir,
            logger=logger,
        )
        try:
            wandb.log({"gll/mean_gll_per_bin": per_bin_mean.tolist()})
        except Exception:  # noqa: BLE001
            pass

        logger.info("GLL summary CSV: %s", out_csv)
        logger.info("GLL heatmap PNG: %s", out_png)
        log_event(
            "gll-localizer-finish",
            {"csv": out_csv, "png": out_png},
            log_dir=cfg.log.log_dir,
            logger=logger,
        )
        try:
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass
        return out_csv


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SpectraMind V50 — GLL Error Localizer"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Hydra-style YAML config"
    )
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
