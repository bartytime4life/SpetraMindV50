"""Calibration pipeline orchestrating the full kill-chain."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .adc_correction import apply_adc_correction
from .calibration_config import CalibrationConfig
from .cosmic_ray_removal import remove_cosmics_sigma_clip
from .dark_current_correction import apply_dark_correction
from .flat_field_correction import apply_flat_field
from .git_env_capture import reproducibility_snapshot
from .jitter_injection import inject_jitter
from .logging_utils import append_debug_md, setup_logging
from .nonlinearity_correction import apply_nonlinearity_correction
from .phase_normalization import phase_normalize
from .photometric_extraction import extract_photometry
from .symbolic_calibration import apply_symbolic_constraints
from .trace_alignment import align_spectral_traces


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


class Calibrator:
    """Orchestrates the calibration kill-chain for a time cube ``(T, H, W)``."""

    def __init__(self, cfg: CalibrationConfig, logger=None, evt=None):
        self.cfg = cfg
        self.logger, self.evt = (logger, evt) if logger and evt else setup_logging()
        self.out_dir = _ensure_dir(self.cfg.io.output_dir)

    def _log_event(self, name: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["event"] = name
        self.evt.log(payload)
        self.logger.info(
            f"{name}: {{" + ", ".join(f"{k}={v}" for k, v in payload.items() if k != 'event') + "}}"
        )

    def run_cube(
        self, cube: np.ndarray, exposure_s: float = 1.0, instrument: str = "AIRS"
    ) -> Dict[str, Any]:
        """Run full calibration on a data cube."""

        cube0 = cube
        cube, jitter = inject_jitter(cube, self.cfg.jitter)
        self._log_event(
            "jitter_injection",
            {
                "enabled": self.cfg.jitter.enabled,
                "mean_shift": float(np.mean(jitter)) if jitter.size else 0.0,
            },
        )

        cube = apply_adc_correction(cube, self.cfg.adc)
        cube = apply_nonlinearity_correction(cube, self.cfg.nonlin)
        cube = apply_dark_correction(cube, exposure_s, self.cfg.dark)
        cube = apply_flat_field(cube, self.cfg.flat)
        cube, cr_mask = remove_cosmics_sigma_clip(cube, self.cfg.cosmic)
        self._log_event("cosmic_ray", {"replaced": int(cr_mask.sum())})

        cube, shifts = align_spectral_traces(cube, self.cfg.align)
        self._log_event(
            "alignment", {"mean_shift": float(np.mean(np.abs(shifts)))}
        )

        flux, flux_err = extract_photometry(cube, self.cfg.photometry)
        flux_norm, meta_norm = phase_normalize(flux, self.cfg.norm)
        self._log_event("normalization", meta_norm)

        axis_map = {"time": 0, "wavelength": 2}
        cube_sym, sym_meta = apply_symbolic_constraints(
            cube, self.cfg.symbolic, axis_map=axis_map
        )
        self._log_event("symbolic_calibration", sym_meta)

        np.save(self.out_dir / f"{instrument}_cube_input.npy", cube0)
        np.save(self.out_dir / f"{instrument}_cube_calibrated.npy", cube_sym)
        np.save(self.out_dir / f"{instrument}_flux.npy", flux_norm)
        np.save(self.out_dir / f"{instrument}_flux_err.npy", flux_err)

        return {
            "instrument": instrument,
            "flux": flux_norm,
            "flux_err": flux_err,
            "cube_calibrated": cube_sym,
            "shifts": shifts,
            "cr_replaced": int(cr_mask.sum()),
            "norm_meta": meta_norm,
            "sym_meta": sym_meta,
        }

    def save_run_manifest(self, meta: Dict[str, Any], name: str = "calibration_run.json") -> Path:
        p = self.out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return p


def run_calibration_one(
    cube_path: str,
    instrument: str,
    cfg: Optional[CalibrationConfig] = None,
    exposure_s: float = 1.0,
) -> Dict[str, Any]:
    cfg = cfg or CalibrationConfig()
    logger, evt = setup_logging()
    append_debug_md(
        "Calibration Start",
        {
            "instrument": instrument,
            "cube_path": cube_path,
            "output_dir": cfg.io.output_dir,
        },
    )
    cal = Calibrator(cfg, logger=logger, evt=evt)
    cube = np.load(cube_path)
    result = cal.run_cube(cube, exposure_s=exposure_s, instrument=instrument)
    manifest = {
        "config": cfg.to_dict(),
        "snapshot": reproducibility_snapshot(),
        "instrument": instrument,
        "cube_path": cube_path,
        "outputs": {
            "flux": str(Path(cfg.io.output_dir) / f"{instrument}_flux.npy"),
            "flux_err": str(Path(cfg.io.output_dir) / f"{instrument}_flux_err.npy"),
            "cube_calibrated": str(
                Path(cfg.io.output_dir) / f"{instrument}_cube_calibrated.npy"
            ),
        },
        "stats": {
            "cr_replaced": result.get("cr_replaced", 0),
            "mean_shift": float(np.mean(np.abs(result.get("shifts", np.array([0]))))),
        },
    }
    cal.save_run_manifest(manifest)
    return result


def run_calibration_batch(
    input_dir: str,
    cfg: Optional[CalibrationConfig] = None,
    exposure_s: float = 1.0,
) -> Dict[str, Any]:
    cfg = cfg or CalibrationConfig()
    logger, evt = setup_logging()
    append_debug_md(
        "Calibration Batch Start",
        {
            "input_dir": input_dir,
            "output_dir": cfg.io.output_dir,
        },
    )
    cal = Calibrator(cfg, logger=logger, evt=evt)
    paths = list(Path(input_dir).glob("*.npy"))
    runs = []
    for p in paths:
        name = p.stem
        inst = "AIRS" if "AIRS" in name.upper() else ("FGS1" if "FGS" in name.upper() else "UNK")
        cube = np.load(p)
        res = cal.run_cube(cube, exposure_s=exposure_s, instrument=inst)
        runs.append({"path": str(p), "instrument": inst, "cr_replaced": res.get("cr_replaced", 0)})
    manifest = {
        "config": cfg.to_dict(),
        "snapshot": reproducibility_snapshot(),
        "batch": runs,
    }
    cal.save_run_manifest(manifest, name="calibration_batch.json")
    return manifest
