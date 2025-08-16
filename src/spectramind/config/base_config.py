"""
Copyright (c) 2025 SpectraMind.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional


@dataclass
class ModelConfig:
    """
    Core model architecture configuration (neuro-symbolic + physics-aware).
    Fields map 1:1 with Hydra group: model/defaults.yaml.
    """

    fgs1_encoder: str = "MambaSSM"  # Long-seq state-space model for FGS1
    airs_encoder: str = "GATConv"  # Edge-aware spectral GNN for AIRS
    decoder_mu: str = "MultiScaleDecoder"  # μ-spectrum multi-resolution head
    decoder_sigma: str = "FlowUncertaintyHead"  # σ head with attention fusion
    latent_dim: int = 512
    # Symbolic constraint weights; each should be in [0, 10] and can be schedule-aware.
    symbolic_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "smoothness": 0.25,
            "nonnegativity": 0.10,
            "molecular_coherence": 0.15,
            "asymmetry": 0.05,
            "photonic_alignment": 0.10,
            "fft_spectral": 0.10,
        }
    )
    # Optional toggles/heads
    use_quantile_decoder: bool = False
    use_diffusion_decoder: bool = False
    # COREL/conformal options
    conformal_corel: bool = True
    corel_coverage: float = 0.90
    # Edge features for AIRS spectral graph
    airs_edge_features: List[str] = field(
        default_factory=lambda: ["d_lambda", "molecule", "region"]
    )
    airs_posenc: str = "fourier"  # {"none","sinusoidal","fourier"}


@dataclass
class TrainingConfig:
    """
    Training hyperparameters & pipeline settings (Hydra group: training/default.yaml).
    """

    seed: int = 42
    device: str = "cuda"  # {"cuda","cpu"}
    precision: str = "amp"  # {"fp32","bf16","amp"}
    batch_size: int = 32
    grad_accum: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 150
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # {"cosine","none","onecycle"}
    checkpoint_interval: int = 5
    resume_from: Optional[str] = None
    # Logging & tracking
    jsonl_events: bool = True
    mlflow_enable: bool = True
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "SpectraMindV50"
    wandb_enable: bool = False
    wandb_project: Optional[str] = None
    # Guardrails
    deterministic: bool = True
    benchmark_cudnn: bool = False
    # Diagnostics toggles (fast/leaderboard modes)
    diagnostics_html: bool = True
    diagnostics_fast: bool = False


@dataclass
class DataConfig:
    """
    Data settings & preprocessing (Hydra group: data/ariel.yaml).
    """

    # Canonical dataset layout; DVC/lakeFS may materialize here.
    root: Path = Path("data")
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    # Normalization/augmentation
    normalize: bool = True
    standardize: bool = False
    augmentations: Dict[str, Any] = field(
        default_factory=lambda: {
            "fgs1_jitter_injection": True,
            "temporal_dropout": 0.05,
            "airs_noise_sigma": 0.002,
        }
    )
    # Masking/calibration knobs
    snr_threshold: float = 0.0
    apply_calibration_pipeline: bool = True
    use_symbolic_masking: bool = True
    molecule_region_masking: bool = True


@dataclass
class PathsConfig:
    """
    Paths & artifacts (Hydra group: paths/config.yaml).
    """

    project_root: Path = Path(".")
    runs_dir: Path = Path("runs")
    logs_dir: Path = Path("logs")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    checkpoints_dir: Path = Path("checkpoints")


@dataclass
class LoggingConfig:
    """
    Logging preferences (Hydra group: logging/default.yaml).
    """

    level: str = "INFO"  # {"DEBUG","INFO","WARNING","ERROR"}
    jsonl_path: Optional[Path] = Path("logs/events.jsonl")
    rotating_log_path: Optional[Path] = Path("logs/spectramind.log")
    rotating_bytes: int = 10 * 1024 * 1024  # 10MB
    rotating_backups: int = 5
    console_rich: bool = True
    color: bool = True


@dataclass
class Config:
    """
    Unified root configuration; mirrors configs/config_v50.yaml and composed groups.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Freeform/extra overrides captured here to preserve CLI-provided context
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Dataclass -> serializable dict."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "Config":
        """
        Construct Config from Mapping (e.g., OmegaConf.to_container result).
        Unknown keys under known groups are passed through via extras.
        """

        def sub(name: str, typ):
            blob = dict(m.get(name, {}))
            # Separate out unknowns to extras
            known_fields = {f.name for f in typ.__dataclass_fields__.values()}
            known = {k: blob[k] for k in blob.keys() if k in known_fields}
            obj = typ(**known)
            extra = {k: blob[k] for k in blob.keys() if k not in known_fields}
            return obj, extra

        model, model_extra = sub("model", ModelConfig)
        training, training_extra = sub("training", TrainingConfig)
        data, data_extra = sub("data", DataConfig)
        paths, paths_extra = sub("paths", PathsConfig)
        logging, logging_extra = sub("logging", LoggingConfig)
        extras = dict(m.get("extras", {}))
        extras.update(
            {
                "model_extra": model_extra,
                "training_extra": training_extra,
                "data_extra": data_extra,
                "paths_extra": paths_extra,
                "logging_extra": logging_extra,
            }
        )
        return cls(
            model=model,
            training=training,
            data=data,
            paths=paths,
            logging=logging,
            extras=extras,
        )

    def merge(self, patch: MutableMapping[str, Any]) -> "Config":
        """
        Return a new Config with keys from patch applied shallowly by group.
        Use Hydra CLI for deep merges; this method is a convenience for runtime tweaks.
        """
        d = self.to_dict()
        for k, v in patch.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], dict):
                d[k].update(v)
            else:
                d[k] = v
        return Config.from_mapping(d)
