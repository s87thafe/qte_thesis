"""Compute standard errors for the treatment effect."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantile_regression import PenalizedQuantileRegression
from .density_estimation import ConditionalDensityEstimator
from .lasso import AdaptiveLasso
from .double_selection import WeightedDoubleSelection


@dataclass
class StandardErrorEstimator:
    """Compute various standard error estimators for the treatment effect."""

    tau: float = 0.5

    def _estimate_v(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate ``theta_tau`` and return ``v = f*(d - X @ theta)``."""
        cde = ConditionalDensityEstimator()
        f_hat = cde.conditional_density_function(X, d, y, self.tau)
        lasso = AdaptiveLasso().fit(X, d, f_hat)
        theta_lasso = lasso.theta_
        v_tilde = f_hat * (d - X @ theta_lasso)
        return v_tilde

    def se_sigma1(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
        """Return ``sigma_{1n}`` as defined in equation (2.15)."""
        v = self._estimate_v(X, d, y)
        return self.tau * (1 - self.tau) / np.mean(v ** 2)

    def se_sigma2(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
        """Return ``sigma_{2n}`` as defined in equation (2.15)."""
        cde = ConditionalDensityEstimator()
        f_hat = cde.conditional_density_function(X, d, y, self.tau)
        qr = PenalizedQuantileRegression(tau=self.tau)
        theta_l1 = qr.fit(X, d, y).theta_
        lasso = AdaptiveLasso().fit(X, d, f_hat)
        theta_lasso = lasso.theta_
        psi_diag = qr.compute_diagonal_psi(d, X)
        lambda_tau = qr.compute_penalty_parameter(d, X, psi_diag, self.tau)
        l1_threshold = lambda_tau / np.sqrt(np.mean(X ** 2, axis=0))
        mask = (
            (np.abs(theta_lasso) > 0)
            | (np.abs(theta_l1[1:]) > l1_threshold.reshape(-1, 1))
        ).ravel()
        M = np.column_stack([d, X[:, mask]])
        Mtil = M * f_hat
        H = (Mtil.T @ Mtil) / y.size
        Hinv = np.linalg.inv(H)
        return self.tau * (1 - self.tau) * Hinv[0, 0]

    def se_sigma3(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
        """Return ``sigma_{3n}`` as defined in equation (2.15)."""
        wds = WeightedDoubleSelection(tau=self.tau).fit(X, d, y)
        alpha_hat = wds.theta_[0, 0]
        beta_hat = wds.theta_[1:]
        cde = ConditionalDensityEstimator()
        f_hat = cde.conditional_density_function(X, d, y, self.tau)
        v = self._estimate_v(X, d, y)
        ind = (y <= d * alpha_hat + X @ beta_hat).astype(float)
        num = np.mean(((ind - self.tau) ** 2) * (v ** 2))
        den = np.mean(f_hat * d * v)
        return num / (den ** 2)