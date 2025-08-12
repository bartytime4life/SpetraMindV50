"""
Calibration kill chain for FGS1 and AIRS data processing.

Implements the 6-step calibration pipeline from the architecture:
1. ADC reversal / bias correction
2. Bad pixel mask & interpolation  
3. Non-linearity correction
4. Dark subtraction
5. Flat-fielding
6. Trace extraction / photometry

All steps are DVC-tracked and log correction magnitudes.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np


class CalibrationStep:
    """Base class for calibration steps."""
    
    def __init__(self, name: str):
        self.name = name
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process calibration step.
        
        Args:
            data: Input data array
            variance: Variance array
            mask: Bad pixel mask (True = good pixel)
            metadata: Metadata dictionary
            
        Returns:
            (calibrated_data, calibrated_variance, updated_mask, step_log)
        """
        raise NotImplementedError
        

class ADCReversalStep(CalibrationStep):
    """ADC reversal / bias correction step."""
    
    def __init__(self, gain: float = 1.0, offset: float = 0.0):
        super().__init__("adc_reversal")
        self.gain = gain
        self.offset = offset
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply ADC reversal: E = (R - offset) * gain"""
        start_time = time.time()
        
        # Statistics before correction
        pre_mean = np.mean(data[mask])
        pre_std = np.std(data[mask])
        
        # Apply correction: E = (R - offset) * gain
        calibrated_data = (data - self.offset) * self.gain
        
        # Propagate variance: Var_E = gain^2 * Var_R
        calibrated_variance = variance * (self.gain ** 2)
        
        # Statistics after correction
        post_mean = np.mean(calibrated_data[mask])
        post_std = np.std(calibrated_data[mask])
        
        step_log = {
            "step": "adc_reversal",
            "t_ms": int((time.time() - start_time) * 1000),
            "params": {"gain": self.gain, "offset": self.offset},
            "delta_stats": {
                "pre_mean": float(pre_mean),
                "pre_std": float(pre_std),
                "post_mean": float(post_mean),
                "post_std": float(post_std),
                "mean_delta": float(post_mean - pre_mean),
                "std_ratio": float(post_std / pre_std) if pre_std > 0 else 1.0
            }
        }
        
        return calibrated_data, calibrated_variance, mask, step_log


class BadPixelMaskStep(CalibrationStep):
    """Bad pixel masking and interpolation step."""
    
    def __init__(
        self, 
        hot_threshold: float = 50000,
        cold_threshold: float = -1000,
        sat_threshold: float = 65000,
        noise_floor: float = 100.0
    ):
        super().__init__("bad_pixel_mask")
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold  
        self.sat_threshold = sat_threshold
        self.noise_floor = noise_floor
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Identify bad pixels and interpolate."""
        start_time = time.time()
        
        original_mask_frac = np.mean(~mask)
        
        # Identify bad pixels
        bad_pixels = (
            (data > self.hot_threshold) |  # Hot pixels
            (data < self.cold_threshold) |  # Cold pixels
            (data > self.sat_threshold) |  # Saturated
            np.isnan(data) |  # NaN values
            np.isinf(data)    # Infinite values
        )
        
        # Update mask
        updated_mask = mask & ~bad_pixels
        
        # Simple interpolation (in practice, use spatio-temporal kernel)
        calibrated_data = data.copy()
        calibrated_variance = variance.copy()
        
        # For simplicity, replace bad pixels with local median
        if np.any(bad_pixels):
            # Use scipy.ndimage.median_filter or custom kernel here
            # For now, simple replacement with nearby good values
            for axis in range(data.ndim):
                calibrated_data = self._interpolate_along_axis(
                    calibrated_data, updated_mask, axis
                )
            
            # Add noise floor to interpolated variance
            calibrated_variance[bad_pixels] += self.noise_floor ** 2
        
        new_mask_frac = np.mean(~updated_mask)
        
        step_log = {
            "step": "bad_pixel_mask",
            "t_ms": int((time.time() - start_time) * 1000),
            "params": {
                "hot_threshold": self.hot_threshold,
                "cold_threshold": self.cold_threshold,
                "sat_threshold": self.sat_threshold,
                "noise_floor": self.noise_floor
            },
            "delta_stats": {
                "original_mask_frac": float(original_mask_frac),
                "new_mask_frac": float(new_mask_frac),
                "bad_pixels_added": int(np.sum(bad_pixels)),
                "total_bad_pixels": int(np.sum(~updated_mask))
            }
        }
        
        return calibrated_data, calibrated_variance, updated_mask, step_log
    
    def _interpolate_along_axis(
        self, 
        data: np.ndarray, 
        mask: np.ndarray, 
        axis: int
    ) -> np.ndarray:
        """Simple interpolation along specified axis."""
        # Placeholder for more sophisticated interpolation
        # In practice, would use scipy.interpolate or custom kernels
        return data


class NonLinearityStep(CalibrationStep):
    """Non-linearity correction step."""
    
    def __init__(self, correction_poly: Optional[List[float]] = None):
        super().__init__("non_linearity")
        # Default quadratic correction: f(x) = x + 1e-6*x^2
        self.correction_poly = correction_poly or [0.0, 1.0, 1e-6]
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply non-linearity correction."""
        start_time = time.time()
        
        # Compute correction: f(E) = Σ_n c_n * E^n
        calibrated_data = np.zeros_like(data)
        jacobian = np.zeros_like(data)  # df/dE for variance propagation
        
        for n, coeff in enumerate(self.correction_poly):
            calibrated_data += coeff * (data ** n)
            if n > 0:
                jacobian += n * coeff * (data ** (n - 1))
        
        # Propagate variance: Var_lin = (df/dE)^2 * Var_E
        calibrated_variance = (jacobian ** 2) * variance
        
        # Compute correction magnitude for logging
        correction = calibrated_data - data
        median_correction_ppm = np.median(correction[mask]) * 1e6 if np.any(mask) else 0.0
        
        step_log = {
            "step": "non_linearity", 
            "t_ms": int((time.time() - start_time) * 1000),
            "params": {"correction_poly": self.correction_poly},
            "delta_stats": {
                "median_correction_ppm": float(median_correction_ppm),
                "max_correction": float(np.max(np.abs(correction[mask]))) if np.any(mask) else 0.0,
                "correction_rms": float(np.sqrt(np.mean(correction[mask]**2))) if np.any(mask) else 0.0
            }
        }
        
        return calibrated_data, calibrated_variance, mask, step_log


class DarkSubtractionStep(CalibrationStep):
    """Dark subtraction step."""
    
    def __init__(self, dark_level: float = 100.0, dark_variance: float = 25.0):
        super().__init__("dark_subtraction")
        self.dark_level = dark_level
        self.dark_variance = dark_variance
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply dark subtraction."""
        start_time = time.time()
        
        pre_mean = np.mean(data[mask]) if np.any(mask) else 0.0
        
        # Apply dark subtraction: E_d = E_lin - Dark
        calibrated_data = data - self.dark_level
        
        # Propagate variance: Var_d = Var_lin + Var_Dark
        calibrated_variance = variance + self.dark_variance
        
        post_mean = np.mean(calibrated_data[mask]) if np.any(mask) else 0.0
        
        step_log = {
            "step": "dark_subtraction",
            "t_ms": int((time.time() - start_time) * 1000),
            "params": {
                "dark_level": self.dark_level,
                "dark_variance": self.dark_variance
            },
            "delta_stats": {
                "pre_mean": float(pre_mean),
                "post_mean": float(post_mean),
                "dark_level": float(self.dark_level)
            }
        }
        
        return calibrated_data, calibrated_variance, mask, step_log


class FlatFieldStep(CalibrationStep):
    """Flat-fielding step."""
    
    def __init__(self, flat_field: Optional[np.ndarray] = None):
        super().__init__("flat_field")
        self.flat_field = flat_field
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply flat-fielding correction."""
        start_time = time.time()
        
        # Use unity flat if none provided
        if self.flat_field is None:
            flat = np.ones_like(data)
        else:
            flat = self.flat_field
            
        # Avoid division by zero
        flat_safe = np.where(flat > 1e-6, flat, 1.0)
        
        # Apply flat-fielding: E_f = E_d / Flat
        calibrated_data = data / flat_safe
        
        # Propagate variance: Var_f = Var_d / Flat^2
        calibrated_variance = variance / (flat_safe ** 2)
        
        flat_norm_stats = {
            "mean": float(np.mean(flat[mask])) if np.any(mask) else 1.0,
            "std": float(np.std(flat[mask])) if np.any(mask) else 0.0,
            "min": float(np.min(flat[mask])) if np.any(mask) else 1.0,
            "max": float(np.max(flat[mask])) if np.any(mask) else 1.0
        }
        
        step_log = {
            "step": "flat_field",
            "t_ms": int((time.time() - start_time) * 1000),
            "params": {"has_custom_flat": self.flat_field is not None},
            "delta_stats": {"flat_norm_stats": flat_norm_stats}
        }
        
        return calibrated_data, calibrated_variance, mask, step_log


class TraceExtractionStep(CalibrationStep):
    """Trace extraction / photometry step."""
    
    def __init__(
        self, 
        aperture_radius: float = 3.0,
        background_annulus: Tuple[float, float] = (5.0, 8.0),
        instrument: str = "fgs1"
    ):
        super().__init__("trace_extraction")
        self.aperture_radius = aperture_radius
        self.background_annulus = background_annulus
        self.instrument = instrument
        
    def process(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extract photometry or spectroscopy."""
        start_time = time.time()
        
        if self.instrument.lower() == "fgs1":
            # FGS1: aperture photometry for white-light curve
            extracted_data, extracted_variance = self._extract_fgs1_photometry(
                data, variance, mask
            )
        else:
            # AIRS: spectral extraction along slit  
            extracted_data, extracted_variance = self._extract_airs_spectroscopy(
                data, variance, mask, metadata
            )
        
        step_log = {
            "step": "trace_extraction",
            "t_ms": int((time.time() - start_time) * 1000),
            "params": {
                "aperture_radius": self.aperture_radius,
                "background_annulus": self.background_annulus,
                "instrument": self.instrument
            },
            "delta_stats": {
                "extracted_shape": list(extracted_data.shape),
                "throughput": float(np.sum(extracted_data > 0) / extracted_data.size)
            }
        }
        
        return extracted_data, extracted_variance, mask, step_log
    
    def _extract_fgs1_photometry(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract FGS1 white-light photometry."""
        # Placeholder for aperture photometry
        # In practice, would use photutils or custom aperture sum
        
        if data.ndim == 3:  # [T, H, W]
            # Sum over spatial dimensions (simple box aperture)
            extracted_flux = np.sum(data * mask, axis=(1, 2))
            extracted_var = np.sum(variance * mask, axis=(1, 2))
        else:
            extracted_flux = data
            extracted_var = variance
            
        return extracted_flux, extracted_var
    
    def _extract_airs_spectroscopy(
        self, 
        data: np.ndarray, 
        variance: np.ndarray, 
        mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract AIRS spectroscopy along slit."""
        # Placeholder for optimal extraction
        # In practice, would use optimal extraction algorithm
        
        if data.ndim == 3:  # [T, H, W] -> [T, 283]
            # Simple extraction along spectral axis (dimension 2)
            n_bins = 283
            extracted_flux = np.zeros((data.shape[0], n_bins))
            extracted_var = np.zeros((data.shape[0], n_bins))
            
            # Bin down wavelength dimension
            for i in range(n_bins):
                if i < data.shape[2]:
                    extracted_flux[:, i] = np.mean(data[:, :, i] * mask[:, :, i], axis=1)
                    extracted_var[:, i] = np.mean(variance[:, :, i] * mask[:, :, i], axis=1)
        else:
            extracted_flux = data
            extracted_var = variance
            
        return extracted_flux, extracted_var


class CalibrationKillChain:
    """
    Main calibration kill chain orchestrator.
    
    Runs the 6-step calibration pipeline and logs all steps.
    """
    
    def __init__(self, instrument: str = "fgs1"):
        self.instrument = instrument
        self.steps = self._create_default_steps()
        self.step_logs = []
        
    def _create_default_steps(self) -> List[CalibrationStep]:
        """Create default calibration steps."""
        steps = [
            ADCReversalStep(gain=1.0, offset=100.0),
            BadPixelMaskStep(),
            NonLinearityStep(),
            DarkSubtractionStep(dark_level=50.0),
            FlatFieldStep(),
            TraceExtractionStep(instrument=self.instrument)
        ]
        return steps
    
    def calibrate(
        self, 
        raw_data: np.ndarray,
        initial_variance: Optional[np.ndarray] = None,
        initial_mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Run full calibration kill chain.
        
        Args:
            raw_data: Raw input data
            initial_variance: Initial variance estimate
            initial_mask: Initial pixel mask (True = good)
            metadata: Additional metadata
            
        Returns:
            (calibrated_data, calibrated_variance, final_mask, step_logs)
        """
        if initial_variance is None:
            # Estimate initial variance from Poisson + read noise
            read_noise = 10.0  # electrons
            initial_variance = np.maximum(raw_data, 0) + read_noise**2
            
        if initial_mask is None:
            initial_mask = np.ones_like(raw_data, dtype=bool)
            
        if metadata is None:
            metadata = {}
        
        # Run calibration steps
        data = raw_data.copy()
        variance = initial_variance.copy()
        mask = initial_mask.copy()
        self.step_logs = []
        
        for step in self.steps:
            data, variance, mask, step_log = step.process(data, variance, mask, metadata)
            self.step_logs.append(step_log)
            
        return data, variance, mask, self.step_logs
    
    def save_logs(self, output_path: str) -> None:
        """Save step logs to JSON file."""
        log_data = {
            "instrument": self.instrument,
            "total_steps": len(self.step_logs),
            "total_time_ms": sum(log.get("t_ms", 0) for log in self.step_logs),
            "step_logs": self.step_logs
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)