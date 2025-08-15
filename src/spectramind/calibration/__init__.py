"""
SpectraMind V50 – Calibration Package

Mission-grade calibration stack for the NeurIPS Ariel Data Challenge.
Includes:
    • photometry extraction from raw FGS1/AIRS frames
    • combined uncertainty calibration (temperature scaling + COREL conformal)
    • calibration checking & visualization
    • CLI entrypoint with structured logging (rotating logs + JSONL events)

This package is Hydra/OMEGACONF friendly but does not require Hydra to run.
"""

from .check_calibration import CalibrationChecker
from .corel_calibration import CORELCalibrator
from .photometry import PhotometryExtractor
from .pipeline import CalibrationPipeline
from .temperature_scaling import TemperatureScaler
from .uncertainty_calibration import UncertaintyCalibrator
from .visualization import CalibrationVisualizer

__all__ = [
    "CalibrationPipeline",
    "PhotometryExtractor",
    "CORELCalibrator",
    "TemperatureScaler",
    "UncertaintyCalibrator",
    "CalibrationChecker",
    "CalibrationVisualizer",
]
