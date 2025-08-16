# src/spectramind/models/tests/test_build_from_cfg.py
# Simple smoke tests to ensure Hydra configs instantiate the full model stack.

from omegaconf import OmegaConf
from spectramind.models import build_from_cfg

def test_build_defaults():
    base = OmegaConf.create({
        "model": {
            "encoder_fgs1": {"_name": "fgs1_mamba", "input_dim": 32, "hidden_dim": 128, "depth": 12, "dropout": 0.1},
            "encoder_airs": {"_name": "airs_gnn", "input_dim": 356, "hidden_dim": 128, "num_layers": 4, "dropout": 0.1},
            "decoder_mu": {"_name": "multi_scale_decoder", "hidden_dim": 128, "output_bins": 283},
            "decoder_sigma": {"_name": "flow_uncertainty_head", "hidden_dim": 128, "output_bins": 283},
            "corel": {"_name": "spectral_corel", "input_dim": 283, "hidden_dim": 128, "output_dim": 283},
        }
    })
    built = build_from_cfg(base)
    assert built.fgs1_encoder is not None
    assert built.airs_encoder is not None
    assert built.mu_decoder is not None
    assert built.sigma_head is not None
    assert built.corel is not None

def test_build_moe_decoder():
    base = OmegaConf.create({
        "model": {
            "encoder_fgs1": {"_name": "fgs1_mamba", "input_dim": 32, "hidden_dim": 128, "depth": 12, "dropout": 0.1},
            "encoder_airs": {"_name": "airs_gnn", "input_dim": 356, "hidden_dim": 128, "num_layers": 4, "dropout": 0.1},
            "decoder_mu": {"_name": "moe_decoder", "hidden_dim": 128, "output_bins": 283, "num_experts": 4},
            "decoder_sigma": {"_name": "flow_uncertainty_head", "hidden_dim": 128, "output_bins": 283},
            "corel": {"_name": "spectral_corel", "input_dim": 283, "hidden_dim": 128, "output_dim": 283},
        }
    })
    built = build_from_cfg(base)
    assert built.mu_decoder is not None
