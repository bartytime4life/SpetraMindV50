"""
Individual physics constraint functions.

These are standalone implementations of the physics constraints that can be used
independently or combined in the SymbolicLossEngine.
"""

from __future__ import annotations

import math
from typing import List, Dict, Optional


def smoothness_loss(
    mu: List[List[float]], 
    weights: Optional[List[float]] = None,
    valid_mask: Optional[List[bool]] = None
) -> float:
    """
    Compute smoothness loss using central second difference.
    
    L_sm = Σ_i w_i * (μ_{i+1} - 2*μ_i + μ_{i-1})^2
    
    Args:
        mu: Predicted spectra [batch_size, 283]
        weights: Per-bin weights [281] (for central difference)
        valid_mask: Valid bin mask [283]
        
    Returns:
        Smoothness loss (scalar)
    """
    if not mu or len(mu[0]) < 3:
        return 0.0
        
    batch_size = len(mu)
    n_bins = len(mu[0])
    
    if weights is None:
        weights = [1.0] * (n_bins - 2)
    if valid_mask is None:
        valid_mask = [True] * n_bins
        
    total_loss = 0.0
    
    for b in range(batch_size):
        mu_b = mu[b]
        for i in range(1, n_bins - 1):
            if valid_mask[i-1] and valid_mask[i] and valid_mask[i+1]:
                # Central second difference
                delta = mu_b[i+1] - 2*mu_b[i] + mu_b[i-1]
                weight = weights[i-1] if i-1 < len(weights) else 1.0
                total_loss += weight * delta * delta
    
    return total_loss / batch_size if batch_size > 0 else 0.0


def non_negativity_loss(mu: List[List[float]], penalty_type: str = "squared") -> float:
    """
    Compute non-negativity constraint loss.
    
    L_nn = Σ_i ReLU(-μ_i)^p where p=1 or p=2
    
    Args:
        mu: Predicted spectra [batch_size, 283]
        penalty_type: "linear" or "squared"
        
    Returns:
        Non-negativity loss (scalar)
    """
    if not mu:
        return 0.0
        
    batch_size = len(mu)
    total_loss = 0.0
    
    for b in range(batch_size):
        for mu_val in mu[b]:
            if mu_val < 0:
                if penalty_type == "squared":
                    total_loss += (-mu_val) ** 2
                else:
                    total_loss += -mu_val
    
    return total_loss / batch_size if batch_size > 0 else 0.0


def molecular_coherence_loss(
    mu: List[List[float]],
    molecule_windows: Dict[str, List[int]],
    templates: Dict[str, List[float]],
    eps: float = 1e-6
) -> float:
    """
    Compute molecular coherence loss (soft implication).
    
    For each molecule m with template t^m and window W_m:
    s^m_i = μ_i / ||μ_{W_m}||_2
    L_coh^m = Σ_{i∈W_m} ReLU(t^m_i - s^m_i)^2
    
    Args:
        mu: Predicted spectra [batch_size, 283]
        molecule_windows: Dict mapping molecule names to bin indices
        templates: Dict mapping molecule names to normalized templates
        eps: Numerical stability constant
        
    Returns:
        Molecular coherence loss (scalar)
    """
    if not mu:
        return 0.0
        
    batch_size = len(mu)
    total_loss = 0.0
    
    for b in range(batch_size):
        mu_b = mu[b]
        
        for mol, indices in molecule_windows.items():
            if mol not in templates:
                continue
                
            template = templates[mol]
            
            # Extract molecule window values
            mu_mol = [mu_b[i] for i in indices if i < len(mu_b)]
            if len(mu_mol) != len(template):
                continue
            
            # Normalize by L2 norm
            norm = math.sqrt(sum(x * x for x in mu_mol))
            if norm < eps:
                continue
                
            s_mol = [x / norm for x in mu_mol]
            
            # Coherence penalty: ReLU(template - normalized_response)^2
            for t_val, s_val in zip(template, s_mol):
                if t_val > s_val:
                    total_loss += (t_val - s_val) ** 2
    
    return total_loss / batch_size if batch_size > 0 else 0.0


def seam_continuity_loss(mu: List[List[float]], seam_idx: int = 141) -> float:
    """
    Compute seam continuity loss.
    
    L_seam = (μ_{s-1} - μ_s)^2
    
    Args:
        mu: Predicted spectra [batch_size, 283]
        seam_idx: Index of the detector seam
        
    Returns:
        Seam continuity loss (scalar)
    """
    if not mu or seam_idx <= 0 or seam_idx >= len(mu[0]):
        return 0.0
        
    batch_size = len(mu)
    total_loss = 0.0
    
    for b in range(batch_size):
        mu_b = mu[b]
        if seam_idx < len(mu_b):
            # Penalty for discontinuity across seam
            diff = mu_b[seam_idx - 1] - mu_b[seam_idx]
            total_loss += diff * diff
    
    return total_loss / batch_size if batch_size > 0 else 0.0


def ratio_penalty_loss(
    mu: List[List[float]],
    molecule_windows: Dict[str, List[int]],
    ratio_constraints: List[tuple],  # [(mol_a, mol_b, r_min, r_max), ...]
    bin_widths: Optional[List[float]] = None,
    eps: float = 1e-6
) -> float:
    """
    Compute chemistry envelope ratio penalties.
    
    For molecules a,b with areas A_a, A_b:
    r = A_a / (A_b + ε)
    L_ratio = ReLU(r_min - r) + ReLU(r - r_max)
    
    Args:
        mu: Predicted spectra [batch_size, 283]
        molecule_windows: Dict mapping molecule names to bin indices
        ratio_constraints: List of (mol_a, mol_b, r_min, r_max) tuples
        bin_widths: Wavelength bin widths [283] (optional)
        eps: Numerical stability constant
        
    Returns:
        Ratio penalty loss (scalar)
    """
    if not mu or not ratio_constraints:
        return 0.0
        
    batch_size = len(mu)
    n_bins = len(mu[0])
    
    if bin_widths is None:
        bin_widths = [0.02] * n_bins  # Default bin width
        
    total_loss = 0.0
    
    for b in range(batch_size):
        mu_b = mu[b]
        
        # Compute band integrals (areas)
        areas = {}
        for mol, indices in molecule_windows.items():
            area = 0.0
            for i in indices:
                if i < len(mu_b) and i < len(bin_widths):
                    area += mu_b[i] * bin_widths[i]
            areas[mol] = area
        
        # Apply ratio constraints
        for mol_a, mol_b, r_min, r_max in ratio_constraints:
            if mol_a in areas and mol_b in areas:
                ratio = areas[mol_a] / (areas[mol_b] + eps)
                
                # ReLU penalties for out-of-range ratios
                if ratio < r_min:
                    total_loss += (r_min - ratio) ** 2
                elif ratio > r_max:
                    total_loss += (ratio - r_max) ** 2
    
    return total_loss / batch_size if batch_size > 0 else 0.0


def quantile_monotonicity_loss(
    q10: List[List[float]], 
    q50: List[List[float]], 
    q90: List[List[float]]
) -> float:
    """
    Compute quantile monotonicity loss.
    
    L_qm = Σ_i [ReLU(q_{10,i} - q_{50,i}) + ReLU(q_{50,i} - q_{90,i})]
    
    Args:
        q10: 10th percentile predictions [batch_size, 283]
        q50: 50th percentile predictions [batch_size, 283] 
        q90: 90th percentile predictions [batch_size, 283]
        
    Returns:
        Quantile monotonicity loss (scalar)
    """
    if not q10 or not q50 or not q90:
        return 0.0
        
    batch_size = len(q10)
    total_loss = 0.0
    
    for b in range(batch_size):
        for i in range(len(q10[b])):
            # Ensure q10 <= q50 <= q90
            if q10[b][i] > q50[b][i]:
                total_loss += (q10[b][i] - q50[b][i]) ** 2
            if q50[b][i] > q90[b][i]:
                total_loss += (q50[b][i] - q90[b][i]) ** 2
    
    return total_loss / batch_size if batch_size > 0 else 0.0