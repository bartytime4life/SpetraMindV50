import logging
from pathlib import Path
from typing import Any, Dict

try:
    import hydra  # optional
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    hydra = None
    DictConfig = dict  # type: ignore

from .check_calibration import CalibrationChecker
from .logging_utils import log_event
from .photometry import PhotometryExtractor
from .uncertainty_calibration import UncertaintyCalibrator

log = logging.getLogger(__name__)


class CalibrationPipeline:
    """
    End-to-end calibration pipeline for SpectraMind V50:
    1) Photometry extraction from raw frames
    2) Uncertainty calibration (temperature scaling + COREL)
    3) Calibration diagnostics (histograms, coverage)
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.photometry = PhotometryExtractor(config.get("photometry", {}))
        self.calibrator = UncertaintyCalibrator(config.get("calibration", {}))
        self.checker = CalibrationChecker(config.get("checker", {}))

    def run(self, raw_data_dir: str, output_dir: str) -> Dict[str, Any]:
        log.info(
            "Calibration pipeline starting",
            extra={"raw_data_dir": raw_data_dir, "output_dir": output_dir},
        )
        log_event(
            "pipeline.start", {"raw_data_dir": raw_data_dir, "output_dir": output_dir}
        )

        out_root = Path(output_dir)
        (out_root / "photometry").mkdir(parents=True, exist_ok=True)
        (out_root / "calibrated").mkdir(parents=True, exist_ok=True)
        (out_root / "diagnostics").mkdir(parents=True, exist_ok=True)

        light_curves = self.photometry.extract(raw_data_dir, out_root / "photometry")
        log_event("photometry.done", {"n_planets": len(light_curves)})

        calibrated = self.calibrator.calibrate(light_curves, out_root / "calibrated")
        log_event(
            "calibration.done",
            {
                "mu_shape": getattr(calibrated.get("mu"), "shape", None),
                "sigma_shape": getattr(calibrated.get("sigma"), "shape", None),
            },
        )

        self.checker.run_checks(calibrated, out_root / "diagnostics")
        log_event("checker.done", {"out_dir": str(out_root / "diagnostics")})

        log.info("Calibration pipeline complete", extra={"output_dir": output_dir})
        log_event("pipeline.complete", {"output_dir": output_dir})
        return calibrated


# Optional Hydra CLI
if hydra is not None:

    @hydra.main(
        version_base=None, config_path="../../configs", config_name="config_v50"
    )
    def main(cfg: DictConfig) -> None:  # pragma: no cover
        pipe = CalibrationPipeline(cfg.get("calibration_pipeline", {}))
        paths = cfg.get("paths", {})
        pipe.run(
            paths.get("raw_data", "data/raw"),
            paths.get("calibrated_output", "outputs/calibrated"),
        )

    if __name__ == "__main__":  # pragma: no cover
        main()
