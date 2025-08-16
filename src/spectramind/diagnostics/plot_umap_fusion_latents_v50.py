"""
UMAP fusion visualization with symbolic links.
"""
import umap
import matplotlib.pyplot as plt

def plot_umap_fusion(latents, symbolic, outdir="diagnostics"):
    reducer = umap.UMAP()
    emb = reducer.fit_transform(latents)
    plt.scatter(emb[:,0], emb[:,1], c=symbolic, cmap="viridis", s=5)
    plt.title("UMAP Fusion Latents × Symbolic Overlays")
    plt.savefig(f"{outdir}/umap_fusion_latents.png")
    plt.close()
    return emb
