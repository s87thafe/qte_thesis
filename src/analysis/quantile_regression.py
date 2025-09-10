"""Penalized quantile regression utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp


@dataclass
class PenalizedQuantileRegression:
    """L1‑penalized quantile regression solved with CVXPY."""

    tau: float = 0.5
    n_sim: int = 500
    gamma: Optional[float] = None
    random_state: int = 0

    theta_: Optional[np.ndarray] = None

    @staticmethod
    def compute_diagonal_psi(d: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute the diagonal scaling matrix :math:`\Psi`.

        Parameters
        ----------
        d:
            Treatment variable of shape ``(n, 1)``.
        X:
            Covariate matrix of shape ``(n, p)``.
        """
        _, p = X.shape
        psi = np.empty(p + 1, dtype=float)
        psi[0] = np.sqrt(np.mean(d ** 2))
        psi[1:] = np.sqrt(np.mean(X ** 2, axis=0))
        return np.diag(psi)

    @staticmethod
    def compute_penalty_parameter(
        d: np.ndarray,
        X: np.ndarray,
        psi_diag: np.ndarray,
        tau: float = 0.5,
        gamma: Optional[float] = None,
        n_sim: int = 500,
        random_state: int = 0,
    ) -> float:
        """Estimate the penalty tuning parameter for quantile regression."""
        n, _ = X.shape
        gamma = gamma if gamma is not None else 0.05 / n
        psi_inv = np.linalg.inv(psi_diag)
        Z = np.column_stack((d, X)).T
        rng = np.random.default_rng(random_state)
        sup_norms = np.empty(n_sim)
        for i in range(n_sim):
            U = rng.random(n)
            weights = tau - (U <= tau).astype(float)
            avg_scores = (weights * Z).mean(axis=1).reshape(-1, 1)
            scaled = psi_inv @ avg_scores
            sup_norms[i] = np.linalg.norm(scaled, np.inf)
        q = np.quantile(sup_norms, 1 - gamma)
        return 1.1 * q * n

    def fit(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> "PenalizedQuantileRegression":
        """Fit the penalized quantile regression model."""
        n, p = X.shape
        theta = cp.Variable((p + 1, 1))
        psi_diag = self.compute_diagonal_psi(d, X)
        lambda_tau = self.compute_penalty_parameter(
            d, X, psi_diag, self.tau, self.gamma, self.n_sim, self.random_state
        )/n
        penalty = lambda_tau * cp.norm(psi_diag @ theta, 1)
        u = y - d * theta[0] - X @ theta[1:]
        check = cp.mean(cp.maximum(self.tau * u, (self.tau - 1) * u))
        cp.Problem(cp.Minimize(check + penalty)).solve()
        self.theta_ = theta.value
        return self

    def predict(self, X: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Predict the conditional quantile for given data."""
        if self.theta_ is None:
            raise ValueError("Model has not been fitted.")
        return d * self.theta_[0] + X @ self.theta_[1:]