from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CalibrationVisualizer:
    """
    Plot helpers for mean μ and σ across bins.
    """

    def plot_mu_sigma(self, mu: np.ndarray, sigma: np.ndarray, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mu_m = mu.mean(axis=0)
        sg_m = sigma.mean(axis=0)
        plt.figure(figsize=(10, 3))
        plt.plot(mu_m, label="μ mean")
        plt.plot(sg_m, label="σ mean")
        plt.legend()
        plt.title("Mean μ and σ across bins")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
