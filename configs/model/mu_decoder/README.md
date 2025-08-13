μ (Mu) Decoder Configs — SpectraMind V50

These Hydra-safe YAMLs define the mean (μ) decoder variants for SpectraMind V50, used to predict the per-bin spectrum (283 bins) for the NeurIPS 2025 Ariel Data Challenge. They are designed to compose cleanly with the dual-encoder architecture:
•FGS1 encoder: Mamba SSM (temporal fidelity, transit-shape alignment)
•AIRS encoder: GNN (wavelength graph w/ edge features)
•σ head (uncertainty): handled elsewhere (Flow/attention fusion); keep μ decoder activation linear or safely clamped.

Key goals: high-fidelity μ prediction with smoothness, physical plausibility, and symbolic compatibility (non-negativity optional at training-time via loss). These config knobs match our training code paths and CLI.

Files
•base.yaml — canonical MLP μ-decoder (sane defaults, challenge-ready).
•multiscale.yaml — multi-scale μ-decoder with pyramid fusion (concat/attention).
•moe.yaml — Mixture-of-Experts μ-decoder (noisy-top‑k router).
•quantile_aux.yaml — optional quantile decoder head for auxiliary training (pinball loss) to improve robustness / tails.

All configs expect Hydra composition under model.decoders.mu_decoder. See inline comments for parameter meanings and recommended values.

Hydra Composition (example)

In your top-level training config (e.g., config_v50.yaml):

defaults:
  - model/mu_decoder@model.decoders.mu_decoder: base
  # or:
  # - model/mu_decoder@model.decoders.mu_decoder: multiscale
  # - model/mu_decoder@model.decoders.mu_decoder: moe
  # - model/mu_decoder@model.decoders.mu_decoder: quantile_aux

Your code should read:
•cfg.model.decoders.mu_decoder.enabled
•cfg.model.decoders.mu_decoder.in_dim
•cfg.model.decoders.mu_decoder.hidden.dims
•cfg.model.decoders.mu_decoder.loss.*
•etc.

Scientific Notes
•Output bins (out_bins=283) match the challenge spectrum bins.
•Keep μ output linear (no final activation), and enforce physical constraints via symbolic loss & post-hoc clamping if needed.
•Smoothness (time/frequency) improves generalization and physical plausibility; see loss.smooth and loss.fft_smooth.
•If using MoE, monitor router aux loss and capacity_factor to avoid underutilization and instability.
•multiscale is recommended if the encoder exposes pyramid features; otherwise prefer base.

Reproducibility & Logging
•All configs include export block for debug exports and log verbosity.
•Pair with CLI flags (--log-level, --dry-run) and write to rotating logs to ensure run traceability.
