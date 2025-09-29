"""Data generating process for estimator simulations."""
from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from typing import Tuple, List


def _powerlaw_coeffs(p: int, c: float, a: float, rng: np.random.Generator, sign: str = "rademacher") -> np.ndarray:
    """Generate approximately sparse coefficients following a power law."""
    j = np.arange(1, p + 1)
    coef = c * j ** (-a)
    if sign == "rademacher":
        coef *= rng.choice([-1, 1], size=p)
    return coef


def sim_dgp(
    n: int = 500,
    p: int = 100,
    rho: float = 0.5,
    alpha: float = 1.0,
    mu: float = 1.0,
    c_beta: float = 0.2,
    a_beta: float = 1.0,
    c_gamma: float = 0.3,
    a_gamma: float = 1.0,
    include_intercept: bool = True,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Simulate an approximately sparse treatment effect model.

    Parameters
    ----------
    n, p: int
        Sample size and number of covariates.
    rho: float
        Autocorrelation parameter for the covariance matrix of ``Z``.
    alpha: float
        True treatment effect.
    mu: float
        Binary indicator for heteroskedasticity in the error term.
    c_beta, a_beta: float
        Parameters controlling sparsity of ``beta``.
    c_gamma, a_gamma: float
        Parameters controlling sparsity of ``gamma``.
    include_intercept: bool
        Whether to prepend an intercept column to the design matrix.
    seed: int | None
        Random seed for reproducibility.

    Returns
    -------
    y, d, X, xcols, beta, gamma
        Simulated outcome, treatment, design matrix, column names, and
        underlying coefficient vectors.
    """
    rng = default_rng(seed)

    # Covariates
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

    # Approximately sparse coefficients via power law decay
    beta = _powerlaw_coeffs(p, c_beta, a_beta, rng, sign="rademacher")
    gamma = _powerlaw_coeffs(p, c_gamma, a_gamma, rng, sign="rademacher")

    # Treatment and outcome
    v = rng.normal(size=n)
    d = Z @ gamma + v
    sigma = np.sqrt(1.0 + mu * d**2)
    u = sigma * rng.normal(size=n)
    y = alpha * d + Z @ beta + u

    if include_intercept:
        X = np.column_stack([np.ones(n), Z])
        xcols = ["const"] + [f"x{i+1}" for i in range(p)]
    else:
        X = Z
        xcols = [f"x{i+1}" for i in range(p)]

    return y, d, X, xcols, beta, gamma