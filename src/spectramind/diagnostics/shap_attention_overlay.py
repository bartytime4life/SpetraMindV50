"""
SHAP × Attention Fusion Overlay
Combines SHAP bin importance with attention weights for diagnostics.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def shap_attention_overlay(shap_values, attention_weights, planet_id, outdir="diagnostics"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fused = shap_values * attention_weights
    plt.plot(fused, label="SHAP × Attention")
    plt.title(f"SHAP × Attention Overlay - {planet_id}")
    plt.legend()
    plt.savefig(f"{outdir}/shap_attention_{planet_id}.png")
    plt.close()
    return fused
