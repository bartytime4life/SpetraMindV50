"""Analog-to-digital conversion (ADC) correction."""

from __future__ import annotations

import numpy as np

from .calibration_config import ADCConfig


def apply_adc_correction(cube: np.ndarray, cfg: ADCConfig) -> np.ndarray:
    """Linear ADC correction: ``out = (cube - bias) * gain``."""

    if not cfg.enabled:
        return cube
    return (cube - cfg.bias) * cfg.gain
