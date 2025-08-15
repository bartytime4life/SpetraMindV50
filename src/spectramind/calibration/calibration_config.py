"""Hydra-safe calibration configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ADCConfig:
    enabled: bool = True
    bias: float = 0.0
    gain: float = 1.0


@dataclass
class NonlinearityConfig:
    enabled: bool = True
    coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])


@dataclass
class DarkConfig:
    enabled: bool = True
    method: Literal["frame", "model"] = "frame"
    frame_path: Optional[str] = None
    temperature_K: Optional[float] = None
    dark_rate_e_per_s: float = 0.0


@dataclass
class FlatConfig:
    enabled: bool = True
    frame_path: Optional[str] = None
    eps: float = 1e-8


@dataclass
class CosmicConfig:
    enabled: bool = True
    sigma: float = 5.0
    time_window: int = 3


@dataclass
class PhotometryConfig:
    method: Literal["aperture", "optimal"] = "aperture"
    aperture_radius_px: int = 6
    background_inner_px: int = 10
    background_outer_px: int = 14


@dataclass
class AlignmentConfig:
    enabled: bool = True
    ref: Literal["FGS1", "AIRS", "auto"] = "auto"
    subpixel: bool = True
    max_shift_px: int = 3


@dataclass
class NormalizationConfig:
    enabled: bool = True
    method: Literal["median-oot", "poly"] = "median-oot"
    poly_order: int = 2
    oot_frac: float = 0.2


@dataclass
class JitterInjectionConfig:
    enabled: bool = False
    std_px: float = 0.2
    seed: int = 1337


@dataclass
class SymbolicCalibrationConfig:
    enabled: bool = True
    nonnegativity: bool = True
    smooth_window: int = 5
    smooth_axis: Literal["time", "wavelength"] = "wavelength"


@dataclass
class SigmaCalibrationConfig:
    temperature_scaling: bool = True
    corel: bool = True


@dataclass
class IOConfig:
    input_dir: Optional[str] = None
    output_dir: str = "artifacts/calibration"
    overwrite: bool = True


@dataclass
class CalibrationConfig:
    adc: ADCConfig = field(default_factory=ADCConfig)
    nonlin: NonlinearityConfig = field(default_factory=NonlinearityConfig)
    dark: DarkConfig = field(default_factory=DarkConfig)
    flat: FlatConfig = field(default_factory=FlatConfig)
    cosmic: CosmicConfig = field(default_factory=CosmicConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    align: AlignmentConfig = field(default_factory=AlignmentConfig)
    norm: NormalizationConfig = field(default_factory=NormalizationConfig)
    jitter: JitterInjectionConfig = field(default_factory=JitterInjectionConfig)
    symbolic: SymbolicCalibrationConfig = field(default_factory=SymbolicCalibrationConfig)
    sigma_calib: SigmaCalibrationConfig = field(default_factory=SigmaCalibrationConfig)
    io: IOConfig = field(default_factory=IOConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_yaml(path: str) -> "CalibrationConfig":
        try:
            from omegaconf import OmegaConf  # type: ignore

            cfg = OmegaConf.load(path)
            data = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            import yaml  # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        def merge(dc: Any, dd: Dict[str, Any]) -> None:
            for k, v in dd.items():
                if hasattr(dc, k):
                    field_val = getattr(dc, k)
                    if isinstance(v, dict) and not isinstance(
                        field_val, (str, int, float, bool, type(None))
                    ):
                        merge(field_val, v)
                    else:
                        setattr(dc, k, v)

        cfg_dc = CalibrationConfig()
        merge(cfg_dc, data or {})
        return cfg_dc
