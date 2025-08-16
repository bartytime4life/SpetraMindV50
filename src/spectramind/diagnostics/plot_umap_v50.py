"""
UMAP Latent Projection
----------------------
Embeds latent vectors into 2D/3D UMAP. Supports symbolic overlays,
interactive HTML export, and cluster CSV logging.
"""

import matplotlib.pyplot as plt
import pandas as pd
import umap


def plot_umap(latents, labels=None, save_png="umap.png", save_csv="umap.csv"):
    reducer = umap.UMAP(random_state=42)
    emb = reducer.fit_transform(latents)
    df = pd.DataFrame(emb, columns=["x", "y"])
    df["label"] = labels
    df.to_csv(save_csv, index=False)
    plt.scatter(df.x, df.y, c=labels, cmap="tab20")
    plt.savefig(save_png)
    plt.close()
    return df
