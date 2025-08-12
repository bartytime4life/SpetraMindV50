# FGS1 Mamba SSM — Hydra configs

Temporal encoder for the FGS1 white-light time series, built on a selective State Space Model (Mamba).
This folder provides a cohesive `base.yaml`, size presets, ablation search grid, and toggles for TorchScript
export and step-latent diagnostics.

**How to compose (examples)**

- Use the base as-is (from your model root):