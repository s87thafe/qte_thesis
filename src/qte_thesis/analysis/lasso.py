"""Adaptive LASSO utilities for double selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp
from scipy.stats import norm


@dataclass
class AdaptiveLasso:
    """Two-step weighted LASSO regression."""

    post_lasso: bool = False
    solver: str = "CLARABEL"

    theta_: Optional[np.ndarray] = None
    theta_post_: Optional[np.ndarray] = None
    n_nonzero_: Optional[int] = None
    n_nonzero_post_: Optional[int] = None

    @staticmethod
    def lambda_lasso(X: np.ndarray, gamma: Optional[float] = None) -> float:
        n, p = X.shape
        gamma = gamma if gamma is not None else 0.05 / n
        q = 1 - gamma / (2 * p)
        inv = norm.ppf(q)
        return 1.1 * np.sqrt(n) * 2 * inv

    @staticmethod
    def penalty_loading_init(f_hat: np.ndarray, X: np.ndarray, d: np.ndarray) -> np.ndarray:
        sup_norm = np.max(np.abs(f_hat * X), axis=0)
        l2_norm = np.sqrt(np.mean(f_hat ** 2 * d ** 2))
        return np.diag(sup_norm * l2_norm)

    @staticmethod
    def weighted_lasso(
        X: np.ndarray,
        d: np.ndarray,
        f_hat: np.ndarray,
        psi_diag: np.ndarray,
        solver: str = "CLARABEL",
    ) -> np.ndarray:
        n, p = X.shape
        theta = cp.Variable((p, 1), name="theta")
        lambda_weighted = AdaptiveLasso.lambda_lasso(X) / n
        penalty = lambda_weighted * cp.norm(psi_diag @ theta, 1)
        resid = d - X @ theta
        weighted_loss = cp.mean(cp.multiply(f_hat**2, resid ** 2))
        prob = cp.Problem(cp.Minimize(weighted_loss + penalty))
        try:
            prob.solve(solver=(solver or cp.CLARABEL))
        except Exception:
            return np.full((p, 1), np.nan)
        if theta.value is None:
            return np.full((p, 1), np.nan)
        return theta.value

    @staticmethod
    def penalty_loading_updated(
        f_hat: np.ndarray, X: np.ndarray, d: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        residuals = f_hat * (d - X @ theta)
        psi_flat = np.sqrt(np.mean(f_hat ** 2 * X ** 2 * residuals ** 2, axis=0))
        return np.diag(psi_flat)

    def fit(self, X: np.ndarray, d: np.ndarray, f_hat: np.ndarray) -> "AdaptiveLasso":
        psi_init = self.penalty_loading_init(f_hat, X, d)
        theta_post = self.weighted_lasso(X, d, f_hat, psi_init, solver=self.solver)
        psi_updated = self.penalty_loading_updated(f_hat, X, d, theta_post)
        theta_lasso = self.weighted_lasso(X, d, f_hat, psi_updated, solver=self.solver)
        self.theta_ = theta_lasso
        self.theta_post_ = theta_post
        self.n_nonzero_ = int(np.count_nonzero(theta_lasso))
        self.n_nonzero_post_ = int(np.count_nonzero(theta_post))
        return self