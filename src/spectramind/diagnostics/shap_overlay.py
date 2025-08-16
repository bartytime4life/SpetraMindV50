"""
SHAP Overlay
------------
Overlays SHAP values on μ spectra. Produces histograms,
entropy scores, JSON metadata, and plots.
"""

import json

import matplotlib.pyplot as plt
import numpy as np


def overlay_shap(
    mu, shap_values, save_json="shap_overlay.json", save_png="shap_overlay.png"
):
    entropy = -np.sum(np.abs(shap_values) * np.log(np.abs(shap_values) + 1e-8))
    meta = dict(entropy=float(entropy), mean_shap=float(np.mean(np.abs(shap_values))))
    with open(save_json, "w") as f:
        json.dump(meta, f, indent=2)
    plt.plot(mu, label="μ")
    plt.bar(range(len(shap_values)), shap_values, alpha=0.4, label="SHAP")
    plt.legend()
    plt.savefig(save_png)
    plt.close()
    return meta
