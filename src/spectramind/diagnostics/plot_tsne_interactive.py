"""
Interactive t-SNE latent visualization (Plotly).
"""
import plotly.express as px

def plot_tsne(latents, labels, out_html="diagnostics/tsne_latents.html"):
    import sklearn.manifold
    tsne = sklearn.manifold.TSNE(n_components=2)
    emb = tsne.fit_transform(latents)
    fig = px.scatter(x=emb[:,0], y=emb[:,1], color=labels.astype(str))
    fig.write_html(out_html)
    return out_html
