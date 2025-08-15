import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class CORELCalibrator:
    """
    Bin-wise conformal coverage (COREL-style) for σ calibration.
    This simplified variant estimates a per-bin residual quantile as a lower bound on σ.
    """

    def __init__(self, config):
        cfg = dict(config or {})
        self.coverage_quantile = float(cfg.get("coverage_quantile", 0.9))
        self.min_sigma = float(cfg.get("min_sigma", 1e-4))

    def fit(
        self, mu_pred: np.ndarray, sigma_pred: np.ndarray, mu_true: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-bin residual quantile (axis=0 → bins).
        Shapes: [N, B] each (N planets, B bins)
        """
        residuals = np.abs(mu_pred - mu_true)
        q = np.quantile(residuals, self.coverage_quantile, axis=0)
        q = np.maximum(q, self.min_sigma)
        log.debug("COREL fit complete", extra={"quantile": self.coverage_quantile})
        return q

    def calibrate(
        self, mu_pred: np.ndarray, sigma_pred: np.ndarray, coverage: np.ndarray
    ):
        """
        Inflate σ to be at least coverage per bin.
        """
        sigma_cal = np.maximum(sigma_pred, coverage[None, :])
        return mu_pred, sigma_cal

    def plot_coverage(self, coverage: np.ndarray, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 3))
        plt.plot(coverage)
        plt.title("COREL Coverage Threshold per Bin")
        plt.xlabel("Bin")
        plt.ylabel("σ_min")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
