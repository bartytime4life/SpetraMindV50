"""
Symbolic physics engine implementing differentiable physics constraints.

This module implements the core symbolic loss terms as described in the architecture:
- Smoothness (2nd derivative L2 penalty)
- Non-negativity constraints
- Molecular coherence (soft implication)
- Seam continuity across detector seams
- Chemistry envelope ratios (CH4:H2O:CO2)
- Quantile monotonicity (if using quantile regression)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path


class SymbolicLossEngine:
    """
    Vectorized, differentiable symbolic physics engine.
    
    Implements the symbolic loss pack from the architecture:
    L_sym = λ_sm*L_sm + λ_nn*L_nn + λ_coh*Σ_m L_coh^m + λ_seam*L_seam + λ_ratio*L_ratio + λ_qm*L_qm
    """
    
    def __init__(
        self,
        lambda_sm: float = 0.1,
        lambda_nn: float = 0.1, 
        lambda_coh: float = 0.1,
        lambda_seam: float = 0.1,
        lambda_ratio: float = 0.1,
        lambda_qm: float = 0.1,
        molecule_windows: Optional[Dict[str, List[int]]] = None,
        seam_idx: int = 141,  # Default seam index
        bin_wavelengths: Optional[List[float]] = None,
        eps: float = 1e-6
    ):
        """
        Initialize symbolic loss engine.
        
        Args:
            lambda_*: Loss term weights
            molecule_windows: Dict mapping molecule names to bin indices
            seam_idx: Detector seam index
            bin_wavelengths: Wavelength for each bin (283 values)
            eps: Numerical stability epsilon
        """
        self.lambda_sm = lambda_sm
        self.lambda_nn = lambda_nn
        self.lambda_coh = lambda_coh
        self.lambda_seam = lambda_seam
        self.lambda_ratio = lambda_ratio
        self.lambda_qm = lambda_qm
        
        self.molecule_windows = molecule_windows or self._default_molecule_windows()
        self.seam_idx = seam_idx
        self.bin_wavelengths = bin_wavelengths or self._default_wavelengths()
        self.eps = eps
        
        # Precompute molecule templates (normalized envelopes)
        self.templates = self._create_templates()
        
        # Precompute smoothness weights (stronger in continuum, weaker near lines)
        self.smoothness_weights = self._create_smoothness_weights()
        
    def _default_molecule_windows(self) -> Dict[str, List[int]]:
        """Default molecule windows (placeholder - should be from config)."""
        return {
            'h2o': list(range(50, 80)) + list(range(200, 230)),  # ~1.4μm and ~6.3μm regions
            'co2': list(range(120, 150)),  # ~4.3μm region  
            'ch4': list(range(90, 120)),   # ~3.3μm region
        }
    
    def _default_wavelengths(self) -> List[float]:
        """Default wavelength grid (placeholder - should be from config)."""
        # Approximate AIRS-CH0 wavelength grid
        return [0.5 + i * 0.02 for i in range(283)]  # 0.5 to ~6.2 μm
    
    def _create_templates(self) -> Dict[str, List[float]]:
        """Create normalized molecular templates."""
        templates = {}
        for mol, indices in self.molecule_windows.items():
            # Simple Gaussian-like template (in practice, use real molecular line data)
            template = []
            center = len(indices) // 2
            for i, _ in enumerate(indices):
                # Gaussian envelope
                val = math.exp(-0.5 * ((i - center) / (len(indices) / 4)) ** 2)
                template.append(val)
            
            # Normalize to unit L2 norm
            norm = math.sqrt(sum(x * x for x in template))
            templates[mol] = [x / (norm + self.eps) for x in template]
        
        return templates
    
    def _create_smoothness_weights(self) -> List[float]:
        """Create smoothness weights (stronger in continuum regions)."""
        weights = [1.0] * 281  # For 2nd derivative, we have 283-2=281 points
        
        # Reduce weights near known molecular features
        for mol, indices in self.molecule_windows.items():
            for idx in indices:
                if 1 <= idx <= 281:  # Valid range for smoothness weights
                    weights[idx - 1] *= 0.5  # Reduce smoothness penalty in molecular regions
        
        return weights
    
    def compute_symbolic_loss(
        self,
        mu: List[List[float]],  # [batch_size, 283]
        sigma: Optional[List[List[float]]] = None,  # [batch_size, 283]  
        q10: Optional[List[List[float]]] = None,  # [batch_size, 283]
        q50: Optional[List[List[float]]] = None,  # [batch_size, 283]
        q90: Optional[List[List[float]]] = None,  # [batch_size, 283]
        valid_mask: Optional[List[bool]] = None,  # [283] - valid bins
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total symbolic loss and individual components.
        
        Args:
            mu: Predicted mean transmission spectra [batch_size, 283]
            sigma: Predicted uncertainties [batch_size, 283] (optional)
            q10, q50, q90: Predicted quantiles [batch_size, 283] (optional)
            valid_mask: Valid bin mask [283] (optional)
            
        Returns:
            (total_loss, component_losses)
        """
        batch_size = len(mu)
        if batch_size == 0:
            return 0.0, {}
        
        if valid_mask is None:
            valid_mask = [True] * 283
            
        # Individual loss components
        losses = {}
        
        # 1. Smoothness loss (2nd derivative penalty)
        if self.lambda_sm > 0:
            losses['smoothness'] = self._smoothness_loss(mu, valid_mask)
        else:
            losses['smoothness'] = 0.0
            
        # 2. Non-negativity loss
        if self.lambda_nn > 0:
            losses['non_negativity'] = self._non_negativity_loss(mu)
        else:
            losses['non_negativity'] = 0.0
            
        # 3. Molecular coherence loss
        if self.lambda_coh > 0:
            losses['molecular_coherence'] = self._molecular_coherence_loss(mu)
        else:
            losses['molecular_coherence'] = 0.0
            
        # 4. Seam continuity loss
        if self.lambda_seam > 0:
            losses['seam_continuity'] = self._seam_continuity_loss(mu)
        else:
            losses['seam_continuity'] = 0.0
            
        # 5. Chemistry ratio loss
        if self.lambda_ratio > 0:
            losses['chemistry_ratios'] = self._chemistry_ratio_loss(mu)
        else:
            losses['chemistry_ratios'] = 0.0
            
        # 6. Quantile monotonicity loss
        if self.lambda_qm > 0 and q10 is not None and q50 is not None and q90 is not None:
            losses['quantile_monotonicity'] = self._quantile_monotonicity_loss(q10, q50, q90)
        else:
            losses['quantile_monotonicity'] = 0.0
        
        # Total weighted loss
        total_loss = (
            self.lambda_sm * losses['smoothness'] +
            self.lambda_nn * losses['non_negativity'] +
            self.lambda_coh * losses['molecular_coherence'] +
            self.lambda_seam * losses['seam_continuity'] +
            self.lambda_ratio * losses['chemistry_ratios'] +
            self.lambda_qm * losses['quantile_monotonicity']
        )
        
        return total_loss, losses
    
    def _smoothness_loss(self, mu: List[List[float]], valid_mask: List[bool]) -> float:
        """Compute smoothness loss (2nd derivative L2 penalty)."""
        total_loss = 0.0
        batch_size = len(mu)
        
        for b in range(batch_size):
            mu_b = mu[b]
            for i in range(1, len(mu_b) - 1):  # Central difference for 2nd derivative
                if valid_mask[i-1] and valid_mask[i] and valid_mask[i+1]:
                    # Second derivative: δ_i = μ_{i+1} - 2*μ_i + μ_{i-1}
                    delta = mu_b[i+1] - 2*mu_b[i] + mu_b[i-1]
                    weight = self.smoothness_weights[i-1] if i-1 < len(self.smoothness_weights) else 1.0
                    total_loss += weight * delta * delta
        
        return total_loss / batch_size if batch_size > 0 else 0.0
    
    def _non_negativity_loss(self, mu: List[List[float]]) -> float:
        """Compute non-negativity penalty."""
        total_loss = 0.0
        batch_size = len(mu)
        
        for b in range(batch_size):
            for mu_val in mu[b]:
                if mu_val < 0:
                    total_loss += (-mu_val) ** 2  # Squared ReLU penalty
        
        return total_loss / batch_size if batch_size > 0 else 0.0
    
    def _molecular_coherence_loss(self, mu: List[List[float]]) -> float:
        """Compute molecular coherence loss (soft implication)."""
        total_loss = 0.0
        batch_size = len(mu)
        
        for b in range(batch_size):
            mu_b = mu[b]
            
            for mol, indices in self.molecule_windows.items():
                template = self.templates[mol]
                
                # Extract molecule window values
                mu_mol = [mu_b[i] for i in indices if i < len(mu_b)]
                if len(mu_mol) != len(template):
                    continue
                
                # Normalize by L2 norm
                norm = math.sqrt(sum(x * x for x in mu_mol))
                if norm < self.eps:
                    continue
                    
                s_mol = [x / norm for x in mu_mol]
                
                # Coherence penalty: ReLU(template - normalized_response)^2
                for i, (t_val, s_val) in enumerate(zip(template, s_mol)):
                    if t_val > s_val:
                        total_loss += (t_val - s_val) ** 2
        
        return total_loss / batch_size if batch_size > 0 else 0.0
    
    def _seam_continuity_loss(self, mu: List[List[float]]) -> float:
        """Compute seam continuity loss."""
        if self.seam_idx <= 0 or self.seam_idx >= 282:
            return 0.0
            
        total_loss = 0.0
        batch_size = len(mu)
        
        for b in range(batch_size):
            mu_b = mu[b]
            if self.seam_idx < len(mu_b):
                # Penalty for discontinuity across seam
                diff = mu_b[self.seam_idx - 1] - mu_b[self.seam_idx]
                total_loss += diff * diff
        
        return total_loss / batch_size if batch_size > 0 else 0.0
    
    def _chemistry_ratio_loss(self, mu: List[List[float]]) -> float:
        """Compute chemistry envelope ratio penalties."""
        total_loss = 0.0
        batch_size = len(mu)
        
        # Example ratio constraints (should be from config)
        ratio_constraints = [
            ('ch4', 'h2o', 0.1, 2.0),  # CH4:H2O ratio between 0.1 and 2.0
            ('co2', 'h2o', 0.5, 5.0),  # CO2:H2O ratio between 0.5 and 5.0
        ]
        
        for b in range(batch_size):
            mu_b = mu[b]
            
            # Compute band integrals (areas)
            areas = {}
            for mol, indices in self.molecule_windows.items():
                area = 0.0
                for i in indices:
                    if i < len(mu_b) and i < len(self.bin_wavelengths):
                        # Approximate bin width
                        dλ = 0.02  # Default bin width
                        if i > 0 and i < len(self.bin_wavelengths) - 1:
                            dλ = (self.bin_wavelengths[i+1] - self.bin_wavelengths[i-1]) / 2
                        area += mu_b[i] * dλ
                areas[mol] = area
            
            # Apply ratio constraints
            for mol_a, mol_b, r_min, r_max in ratio_constraints:
                if mol_a in areas and mol_b in areas:
                    ratio = areas[mol_a] / (areas[mol_b] + self.eps)
                    
                    # ReLU penalties for out-of-range ratios
                    if ratio < r_min:
                        total_loss += (r_min - ratio) ** 2
                    elif ratio > r_max:
                        total_loss += (ratio - r_max) ** 2
        
        return total_loss / batch_size if batch_size > 0 else 0.0
    
    def _quantile_monotonicity_loss(
        self, 
        q10: List[List[float]], 
        q50: List[List[float]], 
        q90: List[List[float]]
    ) -> float:
        """Compute quantile monotonicity loss."""
        total_loss = 0.0
        batch_size = len(q10)
        
        for b in range(batch_size):
            for i in range(len(q10[b])):
                # Ensure q10 <= q50 <= q90
                if q10[b][i] > q50[b][i]:
                    total_loss += (q10[b][i] - q50[b][i]) ** 2
                if q50[b][i] > q90[b][i]:
                    total_loss += (q50[b][i] - q90[b][i]) ** 2
        
        return total_loss / batch_size if batch_size > 0 else 0.0


def create_symbolic_engine_from_config(config_path: Optional[str] = None) -> SymbolicLossEngine:
    """Create symbolic engine from config file."""
    # Placeholder implementation - in practice, load from YAML config
    if config_path and Path(config_path).exists():
        # Would load YAML config here
        pass
    
    return SymbolicLossEngine(
        lambda_sm=0.1,
        lambda_nn=0.1,
        lambda_coh=0.1,
        lambda_seam=0.1,
        lambda_ratio=0.1,
        lambda_qm=0.1
    )