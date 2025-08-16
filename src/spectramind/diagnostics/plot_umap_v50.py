"""
UMAP latent visualization with symbolic overlays.
"""
import umap
import matplotlib.pyplot as plt

def plot_umap(latents, labels, outdir="diagnostics"):
    reducer = umap.UMAP()
    emb = reducer.fit_transform(latents)
    plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="Spectral", s=5)
    plt.title("UMAP Latents with Symbolic Overlays")
    plt.savefig(f"{outdir}/umap_latents.png")
    plt.close()
    return emb
