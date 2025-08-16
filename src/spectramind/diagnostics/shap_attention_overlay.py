"""
SHAP × Attention Overlay
------------------------
Combines SHAP values with attention maps to highlight
joint regions of high attribution and symbolic violations.
"""

import matplotlib.pyplot as plt
import numpy as np


def shap_attention_fusion(shap_vals, attn, save_png="shap_attention.png"):
    fusion = shap_vals * attn
    plt.imshow(fusion[np.newaxis, :], aspect="auto", cmap="viridis")
    plt.colorbar(label="SHAP × Attention")
    plt.savefig(save_png)
    plt.close()
    return fusion
