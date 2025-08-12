"""
COREL Spectral Conformalization for uncertainty calibration.

COREL (Conformal Regression via Error Learning) provides per-bin calibrated
uncertainties by learning conformal scores and applying conformalization.

The algorithm:
1. Compute conformal scores per bin: ε_i = |y_i - μ_i| / σ*_i  
2. Learn quantiles q_i to target coverage 1-α on validation set
3. Calibrated uncertainties: σ'_i = q_i * σ*_i

This provides theoretical coverage guarantees under exchangeability.
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class CORELSpectralConformal:
    """
    COREL Spectral Conformalization for per-bin uncertainty calibration.
    
    Implements conformal prediction with per-bin coverage targeting
    for spectroscopic data with molecule-region awareness.
    """
    
    def __init__(
        self, 
        alpha: float = 0.1,
        molecule_windows: Optional[Dict[str, List[int]]] = None,
        min_coverage: float = 0.8,
        max_coverage: float = 0.99
    ):
        """
        Initialize COREL conformalization.
        
        Args:
            alpha: Miscoverage level (target coverage = 1-α)
            molecule_windows: Dict mapping molecule names to bin indices
            min_coverage: Minimum allowed coverage per bin
            max_coverage: Maximum allowed coverage per bin
        """
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self.molecule_windows = molecule_windows or {}
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        
        # Fitted parameters
        self.quantiles = None  # Per-bin quantiles [283]
        self.is_fitted = False
        
        # Diagnostic information
        self.coverage_stats = {}
        self.score_stats = {}
        
    def fit(
        self,
        sigma_temp: List[List[float]],  # Temperature-scaled uncertainties
        mu: List[List[float]], 
        targets: List[List[float]],
        molecule_aware: bool = True
    ) -> List[float]:
        """
        Fit COREL conformalization on validation data.
        
        Args:
            sigma_temp: Temperature-scaled uncertainties [batch_size, 283]
            mu: Predicted means [batch_size, 283]
            targets: True targets [batch_size, 283]
            molecule_aware: Whether to use molecule-region specific coverage
            
        Returns:
            Per-bin quantiles [283]
        """
        if not sigma_temp or not mu or not targets:
            return []
            
        batch_size = len(sigma_temp)
        n_bins = len(sigma_temp[0]) if sigma_temp else 0
        
        if n_bins == 0:
            return []
        
        # Compute conformal scores per bin
        scores_per_bin = [[] for _ in range(n_bins)]
        
        sigma_min = 1e-4  # Numerical stability
        
        for b in range(batch_size):
            for i in range(n_bins):
                if (i < len(mu[b]) and i < len(targets[b]) and i < len(sigma_temp[b])):
                    s = max(sigma_temp[b][i], sigma_min)
                    m = mu[b][i]
                    y = targets[b][i]
                    
                    # Conformal score: |y - μ| / σ*
                    score = abs(y - m) / s
                    scores_per_bin[i].append(score)
        
        # Compute per-bin quantiles
        self.quantiles = []
        self.coverage_stats = {}
        self.score_stats = {}
        
        for i in range(n_bins):
            scores = scores_per_bin[i]
            if not scores:
                self.quantiles.append(1.0)  # Default quantile
                continue
                
            # Determine target coverage for this bin
            target_cov = self._get_target_coverage(i, molecule_aware)
            
            # Compute empirical quantile
            quantile_level = target_cov
            quantile = self._compute_quantile(scores, quantile_level)
            
            # Clamp to reasonable range
            quantile = max(0.1, min(quantile, 10.0))
            self.quantiles.append(quantile)
            
            # Store diagnostics
            self.coverage_stats[f'bin_{i}'] = {
                'target_coverage': target_cov,
                'n_scores': len(scores),
                'quantile': quantile
            }
            
            if scores:
                self.score_stats[f'bin_{i}'] = {
                    'mean': sum(scores) / len(scores),
                    'std': math.sqrt(sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores)) if len(scores) > 1 else 0.0,
                    'min': min(scores),
                    'max': max(scores),
                    'median': self._compute_quantile(scores, 0.5)
                }
        
        self.is_fitted = True
        return self.quantiles
    
    def _get_target_coverage(self, bin_idx: int, molecule_aware: bool) -> float:
        """
        Get target coverage for a specific bin.
        
        Args:
            bin_idx: Bin index
            molecule_aware: Whether to use molecule-specific coverage
            
        Returns:
            Target coverage for this bin
        """
        if not molecule_aware:
            return self.target_coverage
            
        # Check if bin is in any molecule window
        for mol, indices in self.molecule_windows.items():
            if bin_idx in indices:
                # Could use molecule-specific coverage targets here
                # For now, use same target but could be tuned per molecule
                if mol == 'h2o':
                    return min(self.target_coverage + 0.02, self.max_coverage)  # Slightly higher for water
                elif mol == 'co2':
                    return self.target_coverage
                elif mol == 'ch4':
                    return min(self.target_coverage + 0.01, self.max_coverage)  # Slightly higher for methane
        
        # Default for continuum regions
        return max(self.target_coverage - 0.01, self.min_coverage)
    
    def _compute_quantile(self, values: List[float], level: float) -> float:
        """
        Compute empirical quantile of values.
        
        Args:
            values: List of values
            level: Quantile level (0 to 1)
            
        Returns:
            Empirical quantile
        """
        if not values:
            return 1.0
            
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Use the standard R-6 quantile method (used in conformal prediction)
        index = level * (n + 1)
        
        if index <= 1:
            return sorted_values[0]
        elif index >= n:
            return sorted_values[-1]
        else:
            lower_idx = int(index) - 1
            upper_idx = min(lower_idx + 1, n - 1)
            weight = index - int(index)
            
            return (1 - weight) * sorted_values[lower_idx] + weight * sorted_values[upper_idx]
    
    def transform(self, sigma_temp: List[List[float]]) -> List[List[float]]:
        """
        Apply COREL calibration to temperature-scaled uncertainties.
        
        Args:
            sigma_temp: Temperature-scaled uncertainties [batch_size, 283]
            
        Returns:
            COREL-calibrated uncertainties [batch_size, 283]
        """
        if not self.is_fitted:
            raise ValueError("COREL must be fitted before transform")
            
        if not self.quantiles:
            return sigma_temp
            
        calibrated = []
        for b in range(len(sigma_temp)):
            calibrated_row = []
            for i, s in enumerate(sigma_temp[b]):
                q = self.quantiles[i] if i < len(self.quantiles) else 1.0
                calibrated_row.append(q * s)
            calibrated.append(calibrated_row)
            
        return calibrated
    
    def fit_transform(
        self,
        sigma_temp: List[List[float]],
        mu: List[List[float]], 
        targets: List[List[float]],
        **fit_kwargs
    ) -> List[List[float]]:
        """
        Fit COREL and transform uncertainties.
        
        Args:
            sigma_temp: Temperature-scaled uncertainties [batch_size, 283]
            mu: Predicted means [batch_size, 283]
            targets: True targets [batch_size, 283]
            **fit_kwargs: Additional arguments for fit()
            
        Returns:
            COREL-calibrated uncertainties [batch_size, 283]
        """
        self.fit(sigma_temp, mu, targets, **fit_kwargs)
        return self.transform(sigma_temp)
    
    def get_coverage_report(self) -> Dict[str, any]:
        """
        Get detailed coverage report for diagnostics.
        
        Returns:
            Dictionary with coverage statistics and plots data
        """
        if not self.is_fitted:
            return {}
            
        report = {
            'target_coverage': self.target_coverage,
            'alpha': self.alpha,
            'n_bins': len(self.quantiles) if self.quantiles else 0,
            'coverage_stats': self.coverage_stats,
            'score_stats': self.score_stats
        }
        
        if self.quantiles:
            report['quantile_summary'] = {
                'mean': sum(self.quantiles) / len(self.quantiles),
                'min': min(self.quantiles),
                'max': max(self.quantiles),
                'std': math.sqrt(sum((q - sum(self.quantiles)/len(self.quantiles))**2 for q in self.quantiles) / len(self.quantiles))
            }
        
        # Molecule-specific summaries
        if self.molecule_windows:
            mol_summaries = {}
            for mol, indices in self.molecule_windows.items():
                mol_quantiles = [self.quantiles[i] for i in indices if i < len(self.quantiles)]
                if mol_quantiles:
                    mol_summaries[mol] = {
                        'mean_quantile': sum(mol_quantiles) / len(mol_quantiles),
                        'n_bins': len(mol_quantiles)
                    }
            report['molecule_summaries'] = mol_summaries
            
        return report
    
    def save(self, path: str) -> None:
        """Save fitted COREL parameters to file."""
        import json
        
        data = {
            'quantiles': self.quantiles,
            'is_fitted': self.is_fitted,
            'alpha': self.alpha,
            'target_coverage': self.target_coverage,
            'coverage_stats': self.coverage_stats,
            'score_stats': self.score_stats
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load fitted COREL parameters from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.quantiles = data['quantiles']
        self.is_fitted = data['is_fitted']
        self.alpha = data['alpha']
        self.target_coverage = data['target_coverage']
        self.coverage_stats = data.get('coverage_stats', {})
        self.score_stats = data.get('score_stats', {})


def create_corel_from_config(config_path: Optional[str] = None) -> CORELSpectralConformal:
    """Create COREL instance from config file."""
    # Placeholder - would load from YAML config
    default_molecule_windows = {
        'h2o': list(range(50, 80)) + list(range(200, 230)),
        'co2': list(range(120, 150)),
        'ch4': list(range(90, 120)),
    }
    
    return CORELSpectralConformal(
        alpha=0.1,
        molecule_windows=default_molecule_windows
    )