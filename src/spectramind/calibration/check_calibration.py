import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class CalibrationChecker:
    """
    Evaluate whether predicted σ roughly matches residual magnitudes:
    - Hist residuals vs σ
    - Z-score histogram (residual/σ) ideally ~N(0,1) if calibrated
    """

    def __init__(self, config):
        self.config = dict(config or {})
        self.z_hist_bins = int(self.config.get("z_hist_bins", 60))

    def run_checks(self, calibrated, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        mu = calibrated["mu"]
        sigma = calibrated["sigma"]

        rng = np.random.default_rng(123)
        # Pseudo ground truth again for checker — in real pipeline feed actual y
        mu_true = mu + rng.normal(0.0, 0.005, size=mu.shape).astype(mu.dtype)

        residuals = np.abs(mu - mu_true)
        z = (mu - mu_true) / np.clip(sigma, 1e-6, None)

        # Residual vs σ hist
        plt.figure(figsize=(8, 4))
        plt.hist(residuals.flatten(), bins=60, alpha=0.6, label="|μ - y|")
        plt.hist(sigma.flatten(), bins=60, alpha=0.6, label="σ")
        plt.legend()
        plt.title("Residuals vs Predicted σ")
        plt.tight_layout()
        plt.savefig(out_dir / "residuals_vs_sigma.png", dpi=150)
        plt.close()

        # Z-score histogram
        plt.figure(figsize=(8, 4))
        plt.hist(z.flatten(), bins=self.z_hist_bins, alpha=0.8)
        plt.title("Z-score Histogram ( (μ - y)/σ )")
        plt.tight_layout()
        plt.savefig(out_dir / "z_hist.png", dpi=150)
        plt.close()

        # Simple numeric summaries
        z_mean = float(np.mean(z))
        z_std = float(np.std(z))
        log.info(
            "Calibration check summaries",
            extra={"z_mean": z_mean, "z_std": z_std, "out_dir": str(out_dir)},
        )

        # Save summaries
        with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(f"z_mean: {z_mean:.6f}\n")
            f.write(f"z_std:  {z_std:.6f}\n")
