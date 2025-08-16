"""
SpectraMind V50 Diagnostics Package
===================================

This package contains the full diagnostics and explainability suite
for the NeurIPS 2025 Ariel Data Challenge pipeline. It includes modules
for GLL error localization, symbolic overlays, SHAP fusion, FFT/UMAP/t-SNE
analysis, calibration checking, and dashboard integration.

All modules follow:
- Documentation-first reproducibility protocol:contentReference[oaicite:5]{index=5}
- NASA-grade scientific modeling rigor:contentReference[oaicite:6]{index=6}
- Physics/astrophysics-informed diagnostics:contentReference[oaicite:7]{index=7}
- Domain templates for experiment tracking:contentReference[oaicite:8]{index=8}
- Exoplanet mission context (Ariel 2029):contentReference[oaicite:9]{index=9}

Each module is CLI-ready, Hydra-safe, and dashboard integrated.
"""

__all__ = [
    "gll_error_localizer",
    "symbolic_influence_map",
    "symbolic_loss",
    "shap_overlay",
    "shap_attention_overlay",
    "fft_power_compare",
    "plot_umap_v50",
    "plot_tsne_interactive",
    "generate_diagnostic_summary",
    "generate_html_report",
    "check_calibration",
    "spectral_smoothness_map",
    "analyze_fft_autocorr_mu",
    "generate_fft_symbolic_fusion",
]
