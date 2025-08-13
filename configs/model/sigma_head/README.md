configs/model/sigma_head/ — Uncertainty Head Config Suite (SpectraMind V50)

This directory contains Hydra-safe, mission-grade configuration presets for the σ (uncertainty) decoder head of SpectraMind V50. Each file composes under model.decoders.sigma_head and aligns with our neuro-symbolic, physics-informed pipeline, Kaggle runtime guardrails, and diagnostics dashboard.

Files
• base.yaml — Canonical defaults; safe starting point for most runs.
• gaussian.yaml — Heteroscedastic Gaussian σ head (fast, baseline-strong).
• flow.yaml — Flow-based σ head with attention × symbolic fusion (most expressive).
• evidential.yaml — Normal-Inverse-Gamma evidential uncertainty (calibration-friendly).
• quantile.yaml — Quantile regression head (pinball loss, robust tails).
• ensemble.yaml — Deep-ensemble aggregation policy for σ fusion.
• calibration.yaml — Temperature scaling, COREL GNN, and Conformal Prediction controls.
• kaggle.yaml — Runtime-aware overrides for 9-hour constraint.
• leaderboard.yaml — Tuned for peak leaderboard stability/calibration.
• debug.yaml — Verbose logging/exports for diagnostics dashboard.

Common Schema

All heads share a common envelope:

model:
  decoders:
    sigma_head:
      kind: <gaussian|flow|evidential|quantile|ensemble>
      in_features: 256            # match your decoder neck
      hidden_features: 256
      num_layers: 4
      dropout: 0.10
      activation: "gelu"          # ["relu","gelu","silu"]
      init: "kaiming_uniform"     # ["kaiming_uniform","xavier_uniform"]
      output:
        min_sigma: 1.0e-05
        max_sigma: 1.0
      fusion:
        enable_attention_fuse: false
        attention_heads: 4
        symbolic_overlay_weight: 0.0
      calibration:
        enable: false
        method: "none"            # ["none","temperature","corel","conformal"]

Additional per-head blocks (e.g., flow:, evidential:, quantile:) are defined in their respective files.

Usage

Compose with Hydra:
• Flow head + COREL:
    - model/sigma_head=flow
    - model/sigma_head=calibration

• Gaussian head + temperature scaling:
    - model/sigma_head=gaussian
    - model/sigma_head=calibration

• Evidential head + conformal fallback (held-out):
    - model/sigma_head=evidential
    - model/sigma_head=calibration

• Kaggle lean:
    - model/sigma_head=kaggle

• Leaderboard:
    - model/sigma_head=leaderboard

Notes
• fusion.symbolic_overlay_weight blends symbolic diagnostics (e.g., rule violations) into σ estimation attention fusion for informed uncertainty inflation around suspect bins/regions.
• calibration.corel assumes availability of your SpectralCOREL module (binwise, edge-aware GNN) and its dataset split handling.
• The conformal section uses per-bin or grouped quantile calibration compatible with your spectral_conformal.py.
• output.min_sigma prevents numerically tiny variances; ensure your loss (GLL) clamps/log-transforms consistently.
