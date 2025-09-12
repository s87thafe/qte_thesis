"""Weighted double selection estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp

from .quantile_regression import PenalizedQuantileRegression
from .density_estimation import ConditionalDensityEstimator
from .lasso import AdaptiveLasso


@dataclass
class WeightedDoubleSelection:
    """Compute the weighted double selection estimator."""

    tau: float = 0.5

    theta_: Optional[np.ndarray] = None
    n_nonzero_: Optional[int] = None

    def fit(self, X: np.ndarray, d: np.ndarray, y: np.ndarray) -> "WeightedDoubleSelection":
        qr = PenalizedQuantileRegression(tau=self.tau)
        theta_l1 = qr.fit(X, d, y).theta_
        cde = ConditionalDensityEstimator()
        f_hat = cde.conditional_density_function(X, d, y, self.tau)
        lasso = AdaptiveLasso().fit(X, d, f_hat)
        theta_lasso = lasso.theta_
        psi_diag = qr.compute_diagonal_psi(d, X)
        mu = qr.compute_penalty_parameter(d, X, psi_diag, self.tau)
        l1_threshold = mu / np.sqrt(np.mean(X ** 2, axis=0))
        mask = ((np.abs(theta_lasso) > 0)|(np.abs(theta_l1[1:]) > l1_threshold.reshape(-1, 1))).ravel()
        X_selected = X[:, mask]
        alpha = cp.Variable(name="alpha")
        if X_selected.shape[1] > 0:
            beta = cp.Variable((X_selected.shape[1], 1), name="beta")
            u = y - d * alpha - X_selected @ beta
        else:
            beta = None
            u = y - d * alpha
        check = cp.mean(cp.multiply(f_hat, self.tau*cp.pos(u) + (1 - self.tau)*cp.pos(-u)))
        cp.Problem(cp.Minimize(check)).solve()
        alpha_val = float(alpha.value)
        beta_full = np.zeros((X.shape[1], 1))
        if beta is not None:
            beta_full[mask, 0] = np.ravel(beta.value)
        self.theta_ = np.vstack((np.array([[alpha_val]]), beta_full))
        self.n_nonzero_ = int(np.count_nonzero(beta_full))
        return self

    def predict(self, X: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise ValueError("Estimator has not been fitted.")
        return self.theta_[0] * d + X @ self.theta_[1:]