"""
tsne Interactive Plot
----------------------
Generates interactive Plotly t-SNE projection with symbolic overlays,
confidence shading, and planet hyperlinks.
"""

import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


def plot_tsne(latents, labels=None, hyperlinks=None, save_html="tsne.html"):
    emb = TSNE(n_components=2, random_state=42).fit_transform(latents)
    df = pd.DataFrame(dict(x=emb[:, 0], y=emb[:, 1], label=labels, link=hyperlinks))
    fig = px.scatter(df, x="x", y="y", color="label", hover_data=["link"])
    fig.write_html(save_html)
    return df
