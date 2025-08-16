"""
SpectraMind V50 Diagnostics Package

This package provides the full suite of diagnostic, visualization, and symbolic
overlay tools for the NeurIPS 2025 Ariel Data Challenge. It integrates:
- GLL scoring and error localization
- Symbolic violation predictors (rule-based + neural + fusion)
- SHAP overlays with μ/σ spectra
- UMAP and t-SNE latent projections
- FFT and autocorrelation diagnostics
- HTML dashboard generator with symbolic overlays
- Self-test for reproducibility and integrity
"""
__all__ = [
    "gll_error_localizer",
    "generate_diagnostic_summary",
    "generate_html_report",
    "shap_overlay",
    "shap_attention_overlay",
    "symbolic_influence_map",
    "symbolic_violation_predictor",
    "symbolic_violation_predictor_nn",
    "symbolic_fusion_predictor",
    "analyze_fft_autocorr_mu",
    "generate_fft_symbolic_fusion",
    "plot_umap_v50",
    "plot_tsne_interactive",
    "plot_umap_fusion_latents_v50",
    "selftest_diagnostics",
]
