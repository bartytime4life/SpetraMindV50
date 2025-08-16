# SpectraMind V50 Models

This directory contains the mission-grade modeling components for the NeurIPS 2025 Ariel Data Challenge.

## Components
- `fgs1_mamba.py`: FGS1 encoder (Mamba SSM)
- `airs_gnn.py`: AIRS spectral GNN
- `multi_scale_decoder.py`: μ decoder
- `moe_decoder.py`: Mixture-of-Experts decoder
- `flow_uncertainty_head.py`: σ decoder
- `spectral_corel.py`: COREL calibration GNN
- `base_model.py`: Model base class
- `utils_layers.py`: Utility residual blocks
- `model_registry.py`: Registry for CLI integration

## Usage
All models integrate with Hydra configs and the `spectramind` CLI.

