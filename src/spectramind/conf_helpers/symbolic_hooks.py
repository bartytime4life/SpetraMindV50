from __future__ import annotations


def inject_symbolic_constraints(cfg):
    """
    Ensure default symbolic constraint weights exist in the config.

    Defaults reflect physics-aware + neuro-symbolic priors used in V50.
    """
    if "symbolic" not in cfg:
        cfg.symbolic = {}
    defaults = {
        "smoothness": 1.0,
        "nonnegativity": 1.0,
        "asymmetry": 0.5,
        "fft_suppression": 0.5,
        "photonic_alignment": 1.0,
    }
    for k, v in defaults.items():
        if k not in cfg.symbolic:
            cfg.symbolic[k] = v
    return cfg
