"""
Calibration Checker
-------------------
Evaluates σ vs residuals calibration, plots calibration curves,
saves summary JSON, integrates with dashboard.
"""

import json

import matplotlib.pyplot as plt
import numpy as np


def check_calibration(
    mu, sigma, y, save_json="calibration.json", save_png="calibration.png"
):
    residuals = np.abs(mu - y)
    coverage = np.mean(residuals < sigma)
    summary = dict(coverage=float(coverage))
    with open(save_json, "w") as f:
        json.dump(summary, f, indent=2)
    plt.scatter(sigma, residuals, alpha=0.3)
    plt.xlabel("σ predicted")
    plt.ylabel("|μ - y| residual")
    plt.savefig(save_png)
    plt.close()
    return summary
