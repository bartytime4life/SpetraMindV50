"""
BaseConfig definition for SpectraMind V50.
Supports nested dataclasses for model, training, calibration, diagnostics, symbolic, and env configs.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class BaseConfig:
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    symbolic: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
