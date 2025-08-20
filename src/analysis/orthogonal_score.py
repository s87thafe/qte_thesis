"""Orthogonal score estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar

from .quantile_regression import PenalizedQuantileRegression
from .density_estimation import ConditionalDensityEstimator
from .lasso import AdaptiveLasso


@dataclass
class OrthogonalScoreEstimator:
    """Estimate the treatment effect via an orthogonal score."""

    tau: float = 0.5

    theta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> "OrthogonalScoreEstimator":
        qr = PenalizedQuantileRegression(tau=self.tau)
        theta_l1 = qr.fit(X, d, y).theta_
        psi_diag = qr.compute_diagonal_psi(d, X)
        lambda_tau = qr.compute_penalty_parameter(d, X, psi_diag, self.tau)
        l1_threshold = lambda_tau / np.sqrt(np.mean(X ** 2, axis=0))
        mask = (np.abs(theta_l1[1:]) > l1_threshold.reshape(-1, 1)).ravel()
        X_selected = X[:, mask]
        alpha_tilde = cp.Variable(name="alpha")
        if X_selected.shape[1] > 0:
            beta_tilde = cp.Variable((X_selected.shape[1], 1), name="beta")
            u = y - d * alpha_tilde - X_selected @ beta_tilde
        else:
            beta_tilde = None
            u = y - d * alpha_tilde
        check = cp.mean(cp.maximum(self.tau * u, (self.tau - 1) * u))
        cp.Problem(cp.Minimize(check)).solve()
        beta_tilde_full = np.zeros((X.shape[1], 1))
        if beta_tilde is not None:
            beta_tilde_full[mask, 0] = np.ravel(beta_tilde.value)
        cde = ConditionalDensityEstimator()
        f_hat = cde.conditional_density_function(X, d, y, self.tau)
        lasso = AdaptiveLasso(post_lasso=True).fit(X, d, f_hat)
        theta_post = lasso.theta_post_

        def score(alpha: float) -> float:
            ind = (y <= d * alpha + X @ beta_tilde_full).astype(float)
            psi = self.tau - ind * f_hat * (d - X @ theta_post)
            num = abs(np.mean(psi)) ** 2
            den = np.mean(psi ** 2)
            return np.inf if den == 0 else num / den

        res = minimize_scalar(score)
        alpha_os = res.x
        self.theta_ = np.vstack((np.array([[alpha_os]]), beta_tilde_full))
        return self

    def predict(self, X: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise ValueError("Estimator has not been fitted.")
        return self.theta_[0] * d + X @ self.theta_[1:]