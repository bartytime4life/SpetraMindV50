"""
Pydantic schema definitions for config validation.
"""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    fgs1_encoder: dict
    airs_encoder: dict
    decoder: dict
    symbolic_modules: dict


class TrainingConfig(BaseModel):
    optimizer: dict
    scheduler: dict
    trainer: dict
    loss: dict


class CalibrationConfig(BaseModel):
    temperature_scaling: dict
    corel: dict


class DiagnosticsConfig(BaseModel):
    fft: dict
    umap: dict
    tsne: dict
    shap: dict
    symbolic: dict


class LoggingConfig(BaseModel):
    console: dict
    file: dict
    jsonl: dict
    mlflow: dict
