"""
GLL Error Localizer for SpectraMind V50

Computes per-bin and per-planet GLL errors and generates:
- Heatmaps
- Clustered error overlays
- Symbolic region violation maps
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def compute_gll_error(mu: np.ndarray, sigma: np.ndarray, y_true: np.ndarray):
    """Compute Gaussian log-likelihood per bin."""
    eps = 1e-9
    var = sigma**2 + eps
    return 0.5 * (np.log(2 * np.pi * var) + ((y_true - mu) ** 2) / var)

def localize_errors(mu, sigma, y_true, planet_id: str, outdir: str = "diagnostics"):
    """Generate and save heatmap + JSON for diagnostics."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    gll = compute_gll_error(mu, sigma, y_true)

    # Heatmap
    plt.imshow(gll[np.newaxis, :], aspect="auto", cmap="inferno")
    plt.colorbar(label="GLL Error")
    plt.title(f"GLL Error Heatmap - {planet_id}")
    plt.savefig(f"{outdir}/gll_error_heatmap_{planet_id}.png")
    plt.close()

    # Save JSON
    summary = {"planet_id": planet_id, "mean_gll": float(np.mean(gll))}
    with open(f"{outdir}/gll_error_summary_{planet_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
