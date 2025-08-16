"""
GLL Error Localizer
-------------------
Computes per-bin Generalized Log-Likelihood (GLL) error maps
for μ spectra against ground truth. Supports heatmap visualization,
JSON export, and integration with symbolic overlays.
"""

import json

import matplotlib.pyplot as plt
import numpy as np


def compute_gll_error(mu_pred, mu_true, sigma):
    eps = 1e-8
    return 0.5 * np.log(2 * np.pi * sigma**2 + eps) + ((mu_true - mu_pred) ** 2) / (
        2 * sigma**2 + eps
    )


def localize_errors(
    mu_pred, mu_true, sigma, save_json="gll_errors.json", save_png="gll_heatmap.png"
):
    gll = compute_gll_error(mu_pred, mu_true, sigma)
    with open(save_json, "w") as f:
        json.dump(gll.tolist(), f, indent=2)
    plt.imshow(gll[np.newaxis, :], aspect="auto", cmap="magma")
    plt.colorbar(label="GLL Error")
    plt.savefig(save_png)
    plt.close()
    return gll
