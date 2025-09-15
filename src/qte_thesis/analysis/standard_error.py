"""Compute standard errors for the treatment effect."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import root_scalar

from .quantile_regression import PenalizedQuantileRegression
from .density_estimation import ConditionalDensityEstimator
from .lasso import AdaptiveLasso
from .double_selection import WeightedDoubleSelection
from .orthogonal_score import OrthogonalScoreEstimator


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
        return float(np.sqrt(self.tau * (1 - self.tau) / np.mean(v ** 2)))

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
        val = self.tau * (1 - self.tau) * Hinv[0, 0]
        if not np.isfinite(val) or val <= 0.0:
            return np.nan
        return float(np.sqrt(val))

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
        return float(np.sqrt(num / (den ** 2)))
    
    def ci_wald(
        self, alpha_hat: float, X: np.ndarray, d: np.ndarray, y: np.ndarray, level: float = 0.95, which: str = "sigma3",
    ) -> Tuple[float, float]:
        """
        Wald CI for α_τ: [α̂ ± z_{1-ξ/2} * σ̂ / sqrt(n)], eq. (1.2).
        `which` in {"sigma1","sigma2","sigma3"} chooses σ̂_k^2 per (2.15).
        Returns (lo, hi).
        """
        n = y.size
        var_map = {
            "sigma1": self.se_sigma1,
            "sigma2": self.se_sigma2,
            "sigma3": self.se_sigma3,
        }
        if which not in var_map:
            raise ValueError("which must be one of {'sigma1','sigma2','sigma3'}")
        sigma_hat = var_map[which](X, d, y)
        z = norm.ppf((1-level)/2)
        half = z * sigma_hat / np.sqrt(n)
        return float(alpha_hat - abs(half)), float(alpha_hat + abs(half))

    def ci_score(
            self, X: np.ndarray, d: np.ndarray, y: np.ndarray, level: float = 0.95, which: str = "OrthogonalScoreEstimator"
        ) -> Tuple[float, float]:

        score_map = {
            "WeightedDoubleSelection": WeightedDoubleSelection(self.tau),
            "OrthogonalScoreEstimator": OrthogonalScoreEstimator(self.tau),
        }
        est = score_map[which].fit(X, d, y)
        theta = np.ravel(est.theta_)
        alpha_hat = float(theta[0])
        beta_hat = theta[1:]

        v = self._estimate_v(X, d, y)

        def Ln(alpha: float) -> float:
            ind = (y <= d * alpha + X @ beta_hat).astype(float)
            psi = (self.tau - ind) * v
            m1, m2 = float(np.mean(psi)), float(np.mean(psi**2))
            return 0.0 if m2 <= 1e-12 else (m1 * m1) / m2

        crit = float(chi2.ppf(level, df=1)) 
        F = lambda a: y.size * Ln(a) - crit

        step = 1.0 / np.sqrt(y.size)

        def solve_side(direction: int) -> float:
            a0, f0 = alpha_hat, F(alpha_hat)
            h = step
            for _ in range(60):
                a1 = a0 + direction * h
                f1 = F(a1)
                if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 <= 0:
                    r = root_scalar(F, bracket=(min(a0, a1), max(a0, a1)), method="brentq", xtol=1e-8)
                    return float(r.root) if r.converged else float("nan")
                h *= 1.8
            return float("nan")

        lo = solve_side(-1)
        hi = solve_side(+1)
        return lo, hi
        