"""Conditional density estimation via finite differences."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .quantile_regression import PenalizedQuantileRegression


@dataclass
class ConditionalDensityEstimator:
    """Estimate conditional densities using quantile regression."""

    regressor_cls: type = PenalizedQuantileRegression

    @staticmethod
    def bandwidth_selection(X: np.ndarray, tau: float = 0.5) -> float:
        """Compute the bandwidth used for density estimation."""
        n, _ = X.shape
        return min(n ** (-1 / 6), tau * (1 - tau) / 2)

    def conditional_quantile_function(
        self, X: np.ndarray, d: np.ndarray, y: np.ndarray, u: float
    ) -> np.ndarray:
        """Estimate the conditional quantile :math:`Q(u \mid z_i, d_i)`."""
        reg = self.regressor_cls(tau=u)
        reg.fit(X, d, y)
        return reg.theta_[0] * d + X @ reg.theta_[1:]

    def conditional_density_function(
        self, X: np.ndarray, d: np.ndarray, y: np.ndarray, tau: float
    ) -> np.ndarray:
        """Finite-difference estimate of the conditional density at ``tau``."""
        h = self.bandwidth_selection(X, tau)
        q_plus = self.conditional_quantile_function(X, d, y, tau + h)
        q_minus = self.conditional_quantile_function(X, d, y, tau - h)
        denom = np.clip(q_plus - q_minus, 1e-12, np.inf)
        f_hat = 2 * h / denom
        return np.maximum(f_hat, 0.0)