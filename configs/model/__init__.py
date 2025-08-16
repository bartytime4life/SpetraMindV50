"""
Hydra model configuration package for SpectraMind V50.

This module organizes model component configs for:
- FGS1 Mamba encoder (long sequence state-space model)
- AIRS GNN (spectral correlation graph encoder)
- Multi-scale spectral decoder (μ head)
- Mixture-of-Experts (MoE) decoder
- Flow-based uncertainty head (σ head)
- SpectralCOREL (graph calibration GNN with edge features)

All configs are registered with Hydra using defaults.yaml.
"""
