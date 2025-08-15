import os
from dataclasses import dataclass, field
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class IOConfig:
    pred_dir: str = "artifacts/predictions"
    mu_name: str = "mu.npy"
    sigma_name: str = "sigma.npy"
    y_true: Optional[str] = None
    planet_ids_csv: Optional[str] = None
    bin_meta_csv: Optional[str] = None
    output_dir: str = "outputs/diagnostics"


@dataclass
class LogConfig:
    log_dir: str = os.environ.get("SPECTRAMIND_LOG_DIR", "logs")
    level: str = "INFO"


@dataclass
class SyncConfig:
    mlflow: bool = _env_flag("SPECTRA_MLFLOW", False)
    wandb: bool = _env_flag("SPECTRA_WANDB", False)
    wandb_project: str = os.environ.get("WANDB_PROJECT", "spectramind-v50")


@dataclass
class PlotConfig:
    dpi: int = 120
    html_interactive: bool = True


@dataclass
class SummaryConfig:
    save_png: bool = True
    save_html: bool = True
    embed_tables: bool = True


@dataclass
class DiagnosticsConfig:
    io: IOConfig = field(default_factory=IOConfig)
    log: LogConfig = field(default_factory=LogConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)


def load_config_from_yaml_or_defaults(path: Optional[str]) -> DiagnosticsConfig:
    """Attempt to load configuration from YAML via OmegaConf; fallback to defaults."""
    cfg = DiagnosticsConfig()
    if path is None or not os.path.exists(path):
        return cfg
    try:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

        def update(dc, d):
            for k, v in d.items():
                if hasattr(dc, k):
                    cur = getattr(dc, k)
                    if isinstance(v, dict) and not isinstance(
                        cur, (str, int, float, bool, type(None))
                    ):
                        update(cur, v)
                    else:
                        setattr(dc, k, v)

        update(cfg, data)
        return cfg
    except Exception:  # noqa: BLE001
        return cfg
