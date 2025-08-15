import argparse
import logging
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from ._config import DiagnosticsConfig, load_config_from_yaml_or_defaults
from ._io import ensure_dir, load_preds_truth, save_csv, save_json
from ._logging import capture_env_and_git, get_logger, log_event


def _safe_ppf(p: float) -> float:
    try:
        return float(norm.ppf(p))
    except Exception:  # noqa: BLE001
        import math

        if p <= 0.0:
            return -1e9
        if p >= 1.0:
            return 1e9
        a = [
            -39.6968302866538,
            220.946098424521,
            -275.928510446969,
            138.357751867269,
            -30.6647980661472,
            2.50662827745924,
        ]
        b = [
            -54.4760987982241,
            161.585836858041,
            -155.698979859887,
            66.8013118877197,
            -13.2806815528857,
        ]
        c = [
            -0.00778489400243029,
            -0.322396458041136,
            -2.40075827716184,
            -2.54973253934373,
            4.37466414146497,
            2.93816398269878,
        ]
        d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[
                5
            ] / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if phigh < p:
            q = math.sqrt(-2 * math.log(1 - p))
            return -((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[
                5
            ] / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )


def calibration_metrics(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    """Compute core calibration metrics."""
    z = (y - mu) / (np.maximum(sigma, 1e-6))
    mean_abs_z = float(np.mean(np.abs(z)))
    std_z = float(np.std(z))
    cov_68 = float(np.mean(np.abs(z) <= _safe_ppf(0.84)))
    cov_90 = float(np.mean(np.abs(z) <= _safe_ppf(0.95)))
    cov_95 = float(np.mean(np.abs(z) <= _safe_ppf(0.975)))
    return {
        "mean_abs_z": mean_abs_z,
        "std_z": std_z,
        "cov_68": cov_68,
        "cov_90": cov_90,
        "cov_95": cov_95,
    }


def plot_z_hist(z: np.ndarray, out_png: str, dpi: int = 120) -> None:
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(8, 5), dpi=dpi)
    vals = z.flatten()
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=80, density=True, alpha=0.7, label="z")
    xs = np.linspace(-4, 4, 400)
    try:
        from scipy.stats import norm as _norm

        plt.plot(xs, _norm.pdf(xs), label="N(0,1)", linestyle="--")
    except Exception:  # noqa: BLE001
        pass
    plt.legend()
    plt.title("Z-score Histogram")
    plt.xlabel("z = (y-μ)/σ")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def coverage_by_bin(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, quantile: float
) -> np.ndarray:
    z = (y - mu) / (np.maximum(sigma, 1e-6))
    thr = abs(_safe_ppf((1 + quantile) / 2.0))
    cov = np.mean(np.abs(z) <= thr, axis=0)
    return cov


def run(
    cfg: DiagnosticsConfig,
    run_name: str = "check-calibration",
) -> dict:
    logger = get_logger(level=getattr(logging, cfg.log.level.upper(), logging.INFO))
    capture_env_and_git(cfg.log.log_dir, logger=logger)
    log_event(
        "calibration-start",
        {"pred_dir": cfg.io.pred_dir},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )

    mu, sigma, y_true = load_preds_truth(
        cfg.io.pred_dir, cfg.io.y_true, cfg.io.mu_name, cfg.io.sigma_name
    )
    if y_true is None:
        raise RuntimeError("y_true is required for calibration check.")

    z = (y_true - mu) / (np.maximum(sigma, 1e-6))
    metrics = calibration_metrics(mu, sigma, y_true)
    out_dir = cfg.io.output_dir
    ensure_dir(out_dir)

    plot_z_hist(z, os.path.join(out_dir, "z_hist.png"), dpi=120)
    cov68 = coverage_by_bin(mu, sigma, y_true, 0.68)
    cov95 = coverage_by_bin(mu, sigma, y_true, 0.95)

    df = pd.DataFrame(
        {
            "bin": np.arange(mu.shape[1]),
            "coverage_68": cov68,
            "coverage_95": cov95,
        }
    )
    csv_path = os.path.join(out_dir, "calibration_per_bin.csv")
    save_csv(df, csv_path)

    summary_path = os.path.join(out_dir, "calibration_summary.json")
    save_json(
        {"metrics": metrics, "csv": os.path.basename(csv_path), "z_hist": "z_hist.png"},
        summary_path,
    )

    log_event(
        "calibration-finish",
        {**metrics, "csv": csv_path},
        log_dir=cfg.log.log_dir,
        logger=logger,
    )
    return {"csv": csv_path, "z_hist": os.path.join(out_dir, "z_hist.png")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SpectraMind V50 — σ Calibration Checker"
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config_from_yaml_or_defaults(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
