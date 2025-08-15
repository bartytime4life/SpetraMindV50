import os
import io
import json
import typing as T
import logging

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class Charts:
    """Chart builder with Plotly-first, Matplotlib fallback.
    Returns HTML divs (Plotly) or base64-embedded <img> tags (Matplotlib)."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    # -------------
    # Utilities
    # -------------
    def _mpl_to_img_html(self, fig) -> str:
        """Convert a Matplotlib figure to a base64 <img> tag for embedding."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        import base64
        b64 = base64.b64encode(buf.read()).decode()
        return f'<img alt="chart" style="width:100%;height:auto" src="data:image/png;base64,{b64}"/>'

    def _empty_note(self, text: str) -> str:
        """Return a muted HTML note if data missing."""
        return f'<div class="muted">{text}</div>'

    # -------------
    # Specific Charts
    # -------------
    def gll_heatmap(self, diagnostic_summary: dict) -> str:
        """Heatmap of per-bin GLL (or a proxy) from diagnostic summary.
        Expects diagnostic_summary["bin_gll"] as list or 2D matrix."""
        arr = diagnostic_summary.get("bin_gll")
        if arr is None:
            return self._empty_note("No GLL data available.")
        import numpy as np
        data = np.array(arr)
        if data.ndim == 1:
            data = data[None, :]
        if _HAS_PLOTLY:
            fig = go.Figure(
                data=go.Heatmap(
                    z=data,
                    colorscale="Viridis",
                    colorbar=dict(title="GLL"),
                )
            )
            fig.update_layout(
                margin=dict(l=4, r=4, t=4, b=4),
                height=340,
                template="plotly_dark",
            )
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        elif _HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 3))
            im = ax.imshow(data, aspect="auto", cmap="viridis")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Planet Row")
            fig.colorbar(im, ax=ax, label="GLL")
            return self._mpl_to_img_html(fig)
        else:
            return self._empty_note("Plot backends unavailable.")

    def fft_power(self, fft_summary: dict) -> str:
        """Plot FFT power or smoothed spectra.
        Expects fft_summary like {"freq": [...], "power": [...]} or multiple series."""
        if not fft_summary:
            return self._empty_note("No FFT summary available.")

        def is_multi(d: dict) -> bool:
            return isinstance(d.get("power"), dict)

        if _HAS_PLOTLY:
            fig = go.Figure()
            if is_multi(fft_summary):
                for label, power in fft_summary["power"].items():
                    fig.add_trace(go.Scatter(x=fft_summary.get("freq", []), y=power, mode="lines", name=str(label)))
            else:
                fig.add_trace(go.Scatter(x=fft_summary.get("freq", []), y=fft_summary.get("power", []), mode="lines"))
            fig.update_layout(
                margin=dict(l=4, r=4, t=4, b=4),
                height=300,
                template="plotly_dark",
                xaxis_title="Frequency",
                yaxis_title="Power",
            )
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        elif _HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 3))
            if is_multi(fft_summary):
                for label, power in fft_summary["power"].items():
                    ax.plot(fft_summary.get("freq", []), power, label=str(label))
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.plot(fft_summary.get("freq", []), fft_summary.get("power", []))
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Power")
            return self._mpl_to_img_html(fig)
        else:
            return self._empty_note("Plot backends unavailable.")

    def calibration(self, calibration_summary: dict) -> str:
        """Show calibration scatter/histogram or coverage curve.
        Expects keys like "sigma", "residual", or "coverage_curve" with {(q): coverage}."""
        if not calibration_summary:
            return self._empty_note("No calibration summary available.")

        coverage_curve = calibration_summary.get("coverage_curve")
        if _HAS_PLOTLY:
            fig = go.Figure()
            if isinstance(coverage_curve, dict):
                qs = sorted(coverage_curve.keys())
                cov = [coverage_curve[q] for q in qs]
                fig.add_trace(go.Scatter(x=qs, y=cov, mode="lines+markers", name="Coverage"))
                fig.add_trace(go.Scatter(x=qs, y=qs, mode="lines", name="Ideal", line=dict(dash="dash")))
                fig.update_xaxes(title="Nominal Quantile")
                fig.update_yaxes(title="Empirical Coverage")
            else:
                sig = calibration_summary.get("sigma", [])
                res = calibration_summary.get("residual", [])
                fig.add_trace(go.Scatter(x=sig, y=res, mode="markers", name="|μ - y| vs σ"))
                fig.update_xaxes(title="σ")
                fig.update_yaxes(title="|μ - y|")
            fig.update_layout(margin=dict(l=4, r=4, t=4, b=4), height=300, template="plotly_dark")
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        elif _HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 3))
            if isinstance(coverage_curve, dict):
                qs = sorted(coverage_curve.keys())
                cov = [coverage_curve[q] for q in qs]
                ax.plot(qs, cov, marker="o", label="Coverage")
                ax.plot(qs, qs, linestyle="--", label="Ideal")
                ax.set_xlabel("Nominal Quantile")
                ax.set_ylabel("Empirical Coverage")
                ax.legend(fontsize=8)
            else:
                sig = calibration_summary.get("sigma", [])
                res = calibration_summary.get("residual", [])
                ax.scatter(sig, res, s=8, alpha=0.8)
                ax.set_xlabel("σ")
                ax.set_ylabel("|μ - y|")
            return self._mpl_to_img_html(fig)
        else:
            return self._empty_note("Plot backends unavailable.")

    def umap(self, projection_summary: dict) -> str:
        """UMAP projection scatter with symbolic overlays (if present).
        Expects {"umap": {"x": [...], "y": [...], "color": [...], "label": [...]}}."""
        um = projection_summary.get("umap", {})
        if not um:
            return self._empty_note("UMAP not available.")
        x, y = um.get("x", []), um.get("y", [])
        color = um.get("color", None)
        label = um.get("label", None)
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=5, opacity=0.85, color=color, colorscale="Turbo", showscale=True),
                    text=label,
                )
            )
            fig.update_layout(margin=dict(l=4, r=4, t=4, b=4), height=360, template="plotly_dark", xaxis_title="UMAP-1", yaxis_title="UMAP-2")
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        elif _HAS_MPL:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.scatter(x, y, c=(color if color is not None else "C0"), s=8, alpha=0.85)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            return self._mpl_to_img_html(fig)
        else:
            return self._empty_note("Plot backends unavailable.")

    def tsne(self, projection_summary: dict) -> str:
        """t-SNE projection scatter.
        Expects {"tsne": {"x": [...], "y": [...], "color": [...], "label": [...]}}."""
        ts = projection_summary.get("tsne", {})
        if not ts:
            return self._empty_note("t-SNE not available.")
        x, y = ts.get("x", []), ts.get("y", [])
        color = ts.get("color", None)
        label = ts.get("label", None)
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=5, opacity=0.85, color=color, colorscale="Plasma", showscale=True),
                    text=label,
                )
            )
            fig.update_layout(margin=dict(l=4, r=4, t=4, b=4), height=360, template="plotly_dark", xaxis_title="t-SNE-1", yaxis_title="t-SNE-2")
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        elif _HAS_MPL:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.scatter(x, y, c=(color if color is not None else "C1"), s=8, alpha=0.85)
            ax.set_xlabel("t-SNE-1")
            ax.set_ylabel("t-SNE-2")
            return self._mpl_to_img_html(fig)
        else:
            return self._empty_note("Plot backends unavailable.")

    def symbolic_heatmap(self, symbolic_summary: dict) -> str:
        """Symbolic rule × planet heatmap.
        Expects {"matrix": [[...]], "rule_names": [...], "planet_ids": [...]}."""
        mat = symbolic_summary.get("matrix")
        if mat is None:
            return self._empty_note("Symbolic summary not available.")
        rules = symbolic_summary.get("rule_names", [])
        planets = symbolic_summary.get("planet_ids", [])
        import numpy as np
        data = np.array(mat)
        if _HAS_PLOTLY:
            fig = go.Figure(
                data=go.Heatmap(
                    z=data,
                    colorscale="Inferno",
                    colorbar=dict(title="Violation Magnitude"),
                    x=rules if len(rules) == data.shape[1] else None,
                    y=planets if len(planets) == data.shape[0] else None,
                )
            )
            fig.update_layout(margin=dict(l=4, r=6, t=4, b=4), height=360, template="plotly_dark")
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        elif _HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 3.6))
            im = ax.imshow(data, aspect="auto", cmap="inferno")
            fig.colorbar(im, ax=ax, label="Violation Magnitude")
            ax.set_xlabel("Rules")
            ax.set_ylabel("Planets")
            return self._mpl_to_img_html(fig)
        else:
            return self._empty_note("Plot backends unavailable.")
