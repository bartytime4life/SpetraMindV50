"""
Temperature scaling for uncertainty calibration.

Temperature scaling learns a global temperature parameter τ > 0 that rescales
the predicted uncertainties to minimize validation set Gaussian log-likelihood:

σ* = τ * σ

where τ is learned by minimizing val GLL.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional
from pathlib import Path


class TemperatureScaling:
    """
    Temperature scaling for uncertainty calibration.
    
    Learns a single global temperature parameter to rescale predicted uncertainties
    to improve calibration on validation data.
    """
    
    def __init__(self, initial_temp: float = 1.0, min_temp: float = 0.1, max_temp: float = 10.0):
        """
        Initialize temperature scaling.
        
        Args:
            initial_temp: Initial temperature value
            min_temp: Minimum allowed temperature 
            max_temp: Maximum allowed temperature
        """
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temperature = initial_temp
        self.is_fitted = False
        
    def fit(
        self, 
        sigma: List[List[float]], 
        mu: List[List[float]], 
        targets: List[List[float]],
        max_iter: int = 100,
        lr: float = 0.01,
        tol: float = 1e-6
    ) -> float:
        """
        Fit temperature parameter using gradient descent on validation GLL.
        
        Args:
            sigma: Predicted uncertainties [batch_size, 283]
            mu: Predicted means [batch_size, 283]
            targets: True targets [batch_size, 283]
            max_iter: Maximum optimization iterations
            lr: Learning rate
            tol: Convergence tolerance
            
        Returns:
            Final temperature value
        """
        if not sigma or not mu or not targets:
            return self.temperature
            
        # Use log parameterization for stability
        log_temp = math.log(self.initial_temp)
        
        best_loss = float('inf')
        best_log_temp = log_temp
        
        for iteration in range(max_iter):
            temp = math.exp(log_temp)
            temp = max(self.min_temp, min(temp, self.max_temp))
            
            # Compute GLL with current temperature
            loss, grad = self._compute_gll_and_gradient(sigma, mu, targets, temp)
            
            if loss < best_loss:
                best_loss = loss
                best_log_temp = log_temp
            
            # Check convergence
            if iteration > 0 and abs(prev_loss - loss) < tol:
                break
                
            # Gradient descent step
            log_temp -= lr * grad
            prev_loss = loss
        
        self.temperature = math.exp(best_log_temp)
        self.temperature = max(self.min_temp, min(self.temperature, self.max_temp))
        self.is_fitted = True
        
        return self.temperature
    
    def _compute_gll_and_gradient(
        self, 
        sigma: List[List[float]], 
        mu: List[List[float]], 
        targets: List[List[float]], 
        temp: float
    ) -> Tuple[float, float]:
        """
        Compute Gaussian log-likelihood and gradient w.r.t. log(temperature).
        
        GLL = 0.5 * Σ [log(2π(τσ)²) + (y-μ)²/(τσ)²]
            = 0.5 * Σ [log(2πσ²) + 2*log(τ) + (y-μ)²/(τ²σ²)]
        
        d(GLL)/d(log τ) = Σ [1 - (y-μ)²/(τ²σ²)]
        """
        batch_size = len(sigma)
        n_bins = len(sigma[0]) if sigma else 0
        
        if batch_size == 0 or n_bins == 0:
            return 0.0, 0.0
            
        total_gll = 0.0
        total_grad = 0.0
        count = 0
        
        sigma_min = 1e-4  # Numerical stability
        
        for b in range(batch_size):
            for i in range(n_bins):
                if i < len(mu[b]) and i < len(targets[b]) and i < len(sigma[b]):
                    s = max(sigma[b][i], sigma_min)
                    m = mu[b][i]
                    y = targets[b][i]
                    
                    # Scaled sigma
                    s_scaled = temp * s
                    
                    # Residual
                    residual = y - m
                    
                    # GLL contribution
                    gll_term = 0.5 * (
                        math.log(2 * math.pi * s_scaled * s_scaled) + 
                        (residual * residual) / (s_scaled * s_scaled)
                    )
                    total_gll += gll_term
                    
                    # Gradient contribution (d/d(log τ))
                    grad_term = 1.0 - (residual * residual) / (temp * temp * s * s)
                    total_grad += grad_term
                    
                    count += 1
        
        if count > 0:
            total_gll /= count
            total_grad /= count
        
        return total_gll, total_grad
    
    def transform(self, sigma: List[List[float]]) -> List[List[float]]:
        """
        Apply temperature scaling to uncertainties.
        
        Args:
            sigma: Input uncertainties [batch_size, 283]
            
        Returns:
            Calibrated uncertainties [batch_size, 283]
        """
        if not self.is_fitted:
            raise ValueError("Temperature scaling must be fitted before transform")
            
        calibrated = []
        for b in range(len(sigma)):
            calibrated_row = [self.temperature * s for s in sigma[b]]
            calibrated.append(calibrated_row)
            
        return calibrated
    
    def fit_transform(
        self,
        sigma: List[List[float]], 
        mu: List[List[float]], 
        targets: List[List[float]],
        **fit_kwargs
    ) -> List[List[float]]:
        """
        Fit temperature scaling and transform uncertainties.
        
        Args:
            sigma: Predicted uncertainties [batch_size, 283]
            mu: Predicted means [batch_size, 283]
            targets: True targets [batch_size, 283]
            **fit_kwargs: Additional arguments for fit()
            
        Returns:
            Calibrated uncertainties [batch_size, 283]
        """
        self.fit(sigma, mu, targets, **fit_kwargs)
        return self.transform(sigma)
    
    def save(self, path: str) -> None:
        """Save fitted temperature to file."""
        with open(path, 'w') as f:
            f.write(f"{self.temperature}\n")
            f.write(f"{self.is_fitted}\n")
    
    def load(self, path: str) -> None:
        """Load fitted temperature from file."""
        with open(path, 'r') as f:
            self.temperature = float(f.readline().strip())
            self.is_fitted = f.readline().strip().lower() == 'true'


def simple_temperature_scaling(
    sigma: List[List[float]], 
    mu: List[List[float]], 
    targets: List[List[float]]
) -> float:
    """
    Simple implementation of temperature scaling.
    
    Args:
        sigma: Predicted uncertainties [batch_size, 283]
        mu: Predicted means [batch_size, 283] 
        targets: True targets [batch_size, 283]
        
    Returns:
        Optimal temperature value
    """
    ts = TemperatureScaling()
    return ts.fit(sigma, mu, targets)