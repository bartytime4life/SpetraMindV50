"""
Data loaders for FGS1 and AIRS datasets.

Implements efficient loading and preprocessing of calibrated data
according to the architecture specifications.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Iterator
from pathlib import Path
import numpy as np


class BaseDataLoader:
    """Base class for data loaders."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def list_available_planets(self) -> List[str]:
        """List all available planet IDs."""
        raise NotImplementedError
        
    def load_planet(self, planet_id: str) -> Dict[str, Any]:
        """Load data for a single planet."""
        raise NotImplementedError


class FGS1Loader(BaseDataLoader):
    """
    Loader for FGS1 white-light curve data.
    
    Handles both calibrated frames and extracted features according to
    the architecture data contracts.
    """
    
    def __init__(
        self, 
        data_dir: str,
        load_calibrated: bool = True,
        load_features: bool = True,
        cache_data: bool = False
    ):
        """
        Initialize FGS1 loader.
        
        Args:
            data_dir: Base data directory
            load_calibrated: Whether to load calibrated frames
            load_features: Whether to load extracted features
            cache_data: Whether to cache loaded data in memory
        """
        super().__init__(data_dir)
        self.load_calibrated = load_calibrated
        self.load_features = load_features
        self.cache_data = cache_data
        self._cache = {}
        
        # Set up paths
        self.calibrated_dir = self.data_dir / "data/calibrated/fgs1"
        self.features_dir = self.data_dir / "data/features/fgs1_white"
        
    def list_available_planets(self) -> List[str]:
        """List all available FGS1 planet IDs."""
        planet_ids = set()
        
        # Check calibrated files
        if self.calibrated_dir.exists():
            for file_path in self.calibrated_dir.glob("fgs1_*.npz"):
                planet_id = file_path.stem.replace("fgs1_", "")
                planet_ids.add(planet_id)
                
        # Check feature files
        if self.features_dir.exists():
            for file_path in self.features_dir.glob("fgs1_white_*.npz"):
                planet_id = file_path.stem.replace("fgs1_white_", "")
                planet_ids.add(planet_id)
                
        return sorted(list(planet_ids))
    
    def load_planet(self, planet_id: str) -> Dict[str, Any]:
        """
        Load FGS1 data for a single planet.
        
        Args:
            planet_id: Planet identifier
            
        Returns:
            Dictionary containing loaded data
        """
        if self.cache_data and planet_id in self._cache:
            return self._cache[planet_id]
            
        result = {"planet_id": planet_id}
        
        # Load calibrated frames
        if self.load_calibrated:
            cal_path = self.calibrated_dir / f"fgs1_{planet_id}.npz"
            if cal_path.exists():
                with np.load(cal_path) as data:
                    result["calibrated"] = {
                        "frames": data["frames"],  # [T, H, W]
                        "variance": data["variance"],  # [T, H, W]
                        "mask": data["mask"]  # [T, H, W]
                    }
            else:
                result["calibrated"] = None
                
        # Load extracted features
        if self.load_features:
            feat_path = self.features_dir / f"fgs1_white_{planet_id}.npz"
            if feat_path.exists():
                with np.load(feat_path) as data:
                    result["features"] = {
                        "flux": data["flux"],  # [T]
                        "time": data["time"],  # [T]
                        "centroid": data["centroid"],  # [T, 2]
                        "jitter": data["jitter"]  # [T, 2]
                    }
            else:
                result["features"] = None
        
        if self.cache_data:
            self._cache[planet_id] = result
            
        return result
    
    def load_batch(self, planet_ids: List[str]) -> List[Dict[str, Any]]:
        """Load data for multiple planets."""
        return [self.load_planet(pid) for pid in planet_ids]
    
    def iter_planets(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all available planets."""
        for planet_id in self.list_available_planets():
            yield self.load_planet(planet_id)
            
    def get_time_series_stats(self, planet_id: str) -> Dict[str, Any]:
        """Get time series statistics for a planet."""
        data = self.load_planet(planet_id)
        stats = {"planet_id": planet_id}
        
        if data.get("features") is not None:
            flux = data["features"]["flux"]
            time = data["features"]["time"]
            
            stats.update({
                "n_observations": len(flux),
                "time_span": float(np.max(time) - np.min(time)) if len(time) > 0 else 0.0,
                "flux_mean": float(np.mean(flux)),
                "flux_std": float(np.std(flux)),
                "flux_median": float(np.median(flux)),
                "flux_range": [float(np.min(flux)), float(np.max(flux))],
                "time_sampling": float(np.median(np.diff(time))) if len(time) > 1 else 0.0
            })
            
        return stats


class AIRSLoader(BaseDataLoader):
    """
    Loader for AIRS spectroscopic data.
    
    Handles both calibrated frames and extracted spectral features.
    """
    
    def __init__(
        self,
        data_dir: str,
        load_calibrated: bool = True,
        load_features: bool = True,
        cache_data: bool = False
    ):
        """
        Initialize AIRS loader.
        
        Args:
            data_dir: Base data directory
            load_calibrated: Whether to load calibrated frames
            load_features: Whether to load extracted features
            cache_data: Whether to cache loaded data in memory
        """
        super().__init__(data_dir)
        self.load_calibrated = load_calibrated
        self.load_features = load_features
        self.cache_data = cache_data
        self._cache = {}
        
        # Set up paths
        self.calibrated_dir = self.data_dir / "data/calibrated/airs"
        self.features_dir = self.data_dir / "data/features/airs_bins"
        
    def list_available_planets(self) -> List[str]:
        """List all available AIRS planet IDs."""
        planet_ids = set()
        
        # Check calibrated files
        if self.calibrated_dir.exists():
            for file_path in self.calibrated_dir.glob("airs_*.npz"):
                planet_id = file_path.stem.replace("airs_", "")
                planet_ids.add(planet_id)
                
        # Check feature files  
        if self.features_dir.exists():
            for file_path in self.features_dir.glob("airs_bins_*.npz"):
                planet_id = file_path.stem.replace("airs_bins_", "")
                planet_ids.add(planet_id)
                
        return sorted(list(planet_ids))
    
    def load_planet(self, planet_id: str) -> Dict[str, Any]:
        """
        Load AIRS data for a single planet.
        
        Args:
            planet_id: Planet identifier
            
        Returns:
            Dictionary containing loaded data
        """
        if self.cache_data and planet_id in self._cache:
            return self._cache[planet_id]
            
        result = {"planet_id": planet_id}
        
        # Load calibrated frames
        if self.load_calibrated:
            cal_path = self.calibrated_dir / f"airs_{planet_id}.npz"
            if cal_path.exists():
                with np.load(cal_path, allow_pickle=True) as data:
                    result["calibrated"] = {
                        "frames": data["frames"],  # [T, H, W]
                        "variance": data["variance"],  # [T, H, W]
                        "mask": data["mask"],  # [T, H, W]
                        "trace_meta": data["trace_meta"].item() if "trace_meta" in data else {}
                    }
            else:
                result["calibrated"] = None
                
        # Load extracted features (spectral bins)
        if self.load_features:
            feat_path = self.features_dir / f"airs_bins_{planet_id}.npz"
            if feat_path.exists():
                with np.load(feat_path) as data:
                    result["features"] = {
                        "flux": data["flux"],  # [T, 283]
                        "time": data["time"]  # [T]
                    }
            else:
                result["features"] = None
        
        if self.cache_data:
            self._cache[planet_id] = result
            
        return result
    
    def load_batch(self, planet_ids: List[str]) -> List[Dict[str, Any]]:
        """Load data for multiple planets."""
        return [self.load_planet(pid) for pid in planet_ids]
    
    def iter_planets(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all available planets."""
        for planet_id in self.list_available_planets():
            yield self.load_planet(planet_id)
            
    def get_spectral_stats(self, planet_id: str) -> Dict[str, Any]:
        """Get spectral statistics for a planet."""
        data = self.load_planet(planet_id)
        stats = {"planet_id": planet_id}
        
        if data.get("features") is not None:
            flux = data["features"]["flux"]  # [T, 283]
            time = data["features"]["time"]
            
            # Overall statistics
            stats.update({
                "n_observations": flux.shape[0],
                "n_spectral_bins": flux.shape[1],
                "time_span": float(np.max(time) - np.min(time)) if len(time) > 0 else 0.0,
                "time_sampling": float(np.median(np.diff(time))) if len(time) > 1 else 0.0
            })
            
            # Per-bin statistics
            mean_spectrum = np.mean(flux, axis=0)  # [283]
            std_spectrum = np.std(flux, axis=0)  # [283]
            
            stats.update({
                "mean_spectrum": mean_spectrum.tolist(),
                "std_spectrum": std_spectrum.tolist(),
                "spectrum_range": [float(np.min(mean_spectrum)), float(np.max(mean_spectrum))],
                "snr_median": float(np.median(mean_spectrum / (std_spectrum + 1e-10)))
            })
            
        return stats
    
    def get_molecule_windows(self) -> Dict[str, List[int]]:
        """Get default molecule window definitions."""
        # These should come from config in practice
        return {
            'h2o': list(range(50, 80)) + list(range(200, 230)),  # ~1.4μm and ~6.3μm
            'co2': list(range(120, 150)),  # ~4.3μm
            'ch4': list(range(90, 120)),   # ~3.3μm
        }


class MultiModalLoader:
    """
    Multi-modal loader that combines FGS1 and AIRS data.
    
    Provides synchronized loading of both instruments for joint modeling.
    """
    
    def __init__(
        self,
        data_dir: str,
        load_calibrated: bool = False,  # Usually just features for modeling
        load_features: bool = True,
        cache_data: bool = False
    ):
        """
        Initialize multi-modal loader.
        
        Args:
            data_dir: Base data directory
            load_calibrated: Whether to load calibrated frames
            load_features: Whether to load extracted features
            cache_data: Whether to cache loaded data
        """
        self.fgs1_loader = FGS1Loader(
            data_dir, load_calibrated, load_features, cache_data
        )
        self.airs_loader = AIRSLoader(
            data_dir, load_calibrated, load_features, cache_data
        )
        
    def list_available_planets(self) -> List[str]:
        """List planets available in both FGS1 and AIRS."""
        fgs1_planets = set(self.fgs1_loader.list_available_planets())
        airs_planets = set(self.airs_loader.list_available_planets())
        return sorted(list(fgs1_planets & airs_planets))  # Intersection
    
    def load_planet(self, planet_id: str) -> Dict[str, Any]:
        """
        Load combined FGS1 + AIRS data for a planet.
        
        Args:
            planet_id: Planet identifier
            
        Returns:
            Combined data dictionary
        """
        fgs1_data = self.fgs1_loader.load_planet(planet_id)
        airs_data = self.airs_loader.load_planet(planet_id)
        
        return {
            "planet_id": planet_id,
            "fgs1": fgs1_data,
            "airs": airs_data
        }
    
    def load_batch(self, planet_ids: List[str]) -> List[Dict[str, Any]]:
        """Load data for multiple planets."""
        return [self.load_planet(pid) for pid in planet_ids]
    
    def iter_planets(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all available planets."""
        for planet_id in self.list_available_planets():
            yield self.load_planet(planet_id)
            
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the entire dataset."""
        available_planets = self.list_available_planets()
        
        summary = {
            "total_planets": len(available_planets),
            "fgs1_only": len(set(self.fgs1_loader.list_available_planets()) - 
                            set(available_planets)),
            "airs_only": len(set(self.airs_loader.list_available_planets()) - 
                            set(available_planets)),
            "both_instruments": len(available_planets)
        }
        
        if available_planets:
            # Sample a few planets for stats
            sample_planets = available_planets[:min(5, len(available_planets))]
            sample_stats = []
            
            for planet_id in sample_planets:
                fgs1_stats = self.fgs1_loader.get_time_series_stats(planet_id)
                airs_stats = self.airs_loader.get_spectral_stats(planet_id)
                sample_stats.append({
                    "planet_id": planet_id,
                    "fgs1": fgs1_stats,
                    "airs": airs_stats
                })
                
            summary["sample_stats"] = sample_stats
            
        return summary


def create_data_loaders(
    data_dir: str,
    mode: str = "multimodal",
    **kwargs
) -> Union[FGS1Loader, AIRSLoader, MultiModalLoader]:
    """
    Factory function to create appropriate data loader.
    
    Args:
        data_dir: Base data directory
        mode: Loader type ('fgs1', 'airs', 'multimodal')
        **kwargs: Additional arguments for loader
        
    Returns:
        Configured data loader
    """
    if mode.lower() == "fgs1":
        return FGS1Loader(data_dir, **kwargs)
    elif mode.lower() == "airs":
        return AIRSLoader(data_dir, **kwargs)
    elif mode.lower() == "multimodal":
        return MultiModalLoader(data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown loader mode: {mode}. Use 'fgs1', 'airs', or 'multimodal'.")