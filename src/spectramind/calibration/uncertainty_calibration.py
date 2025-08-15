import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .corel_calibration import CORELCalibrator

log = logging.getLogger(__name__)


class UncertaintyCalibrator:
    """
    Combined uncertainty calibration stage. For this production-safe baseline:
    - Treat input light curves as μ predictions across time/bins
    - Synthesize a pseudo ground truth by adding tiny Gaussian to emulate held-out target
    - Fit per-bin COREL coverage & inflate σ accordingly
    """

    def __init__(self, config):
        cfg = dict(config or {})
        self.corel = CORELCalibrator(cfg.get("corel", {}))
        self.default_sigma = float(
            cfg.get("default_sigma", 0.02)
        )  # baseline σ if none provided

    def calibrate(
        self, light_curves: Dict[str, np.ndarray], out_dir: Path
    ) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Stack as [N, B]
        planets = sorted(list(light_curves.keys()))
        mu_pred = np.stack([light_curves[p] for p in planets], axis=0)
        # Pseudo "true" for wiring tests (in real pipeline, supply actual targets)
        rng = np.random.default_rng(42)
        mu_true = mu_pred + rng.normal(0.0, 0.005, size=mu_pred.shape).astype(
            mu_pred.dtype
        )
        sigma_pred = np.full_like(mu_pred, self.default_sigma, dtype=mu_pred.dtype)

        coverage = self.corel.fit(mu_pred, sigma_pred, mu_true)
        mu_cal, sigma_cal = self.corel.calibrate(mu_pred, sigma_pred, coverage)

        np.save(out_dir / "mu.npy", mu_cal)
        np.save(out_dir / "sigma.npy", sigma_cal)
        np.save(out_dir / "coverage.npy", coverage)
        self.corel.plot_coverage(coverage, out_dir / "coverage.png")

        log.info(
            "Uncertainty calibration complete",
            extra={"N": mu_cal.shape[0], "B": mu_cal.shape[1], "out": str(out_dir)},
        )
        return {
            "mu": mu_cal,
            "sigma": sigma_cal,
            "planets": planets,
            "coverage": coverage,
        }
