from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Dict, Any

import numpy as np
import pandas as pd

from .dgp import sim_dgp
from qte_thesis.analysis.double_selection import WeightedDoubleSelection
from qte_thesis.analysis.orthogonal_score import OrthogonalScoreEstimator
from qte_thesis.analysis.standard_error import StandardErrorEstimator


@dataclass
class SimulationConfig:
    """Configuration for the Monte Carlo experiment.

    Parameters
    ----------
    n_vals, p_vals, taus : iterable of int or float
        Collections of sample sizes, covariate counts and quantile levels to
        iterate over.
    mu_vals : iterable of float, optional
        Values for the heteroscedasticity parameter ``mu``.  Setting ``mu=0``
        yields a homoscedastic design.  The default compares homoscedastic and
        heteroscedastic designs ``(0.0, 1.0)``.
    n_sim : int, optional
        Number of Monte Carlo replications for each scenario.
    alpha_true : float, optional
        True treatment effect used in the data generating process.
    seed : int, optional
        Base seed for reproducibility; a different seed is drawn for each
        replication to avoid perfect correlation across scenarios.
    """

    n_vals: Iterable[int]
    p_vals: Iterable[int]
    taus: Iterable[float]
    mu_vals: Iterable[float] = (0.0, 1.0)
    n_sim: int = 100
    alpha_true: float = 1.0
    seed: int | None = None



def run_simulation(cfg: SimulationConfig) -> pd.DataFrame:
    """Run the Monte Carlo study and return a dataframe of summary metrics."""

    rng = np.random.default_rng(cfg.seed)

    rows: List[Dict[str, Any]] = []

    # Iterate over all design parameters
    for n, p, tau, mu in product(cfg.n_vals, cfg.p_vals, cfg.taus, cfg.mu_vals):
        for rep in range(cfg.n_sim):
            seed = int(rng.integers(1_000_000_000))
            y, d, X, *_ = sim_dgp(n=n, p=p, mu=mu, alpha=cfg.alpha_true, seed=seed)
            y = y.reshape(-1, 1)
            d = d.reshape(-1, 1)

            for est_name, Est in [
                ("DS", WeightedDoubleSelection),
                ("OS", OrthogonalScoreEstimator),
            ]:
                est = Est(tau=tau).fit(X, d, y)
                alpha_hat = float(est.theta_[0, 0])
                support = int(est.n_nonzero_)

                se_est = StandardErrorEstimator(tau=tau)

                # Wald intervals for each of the three SEs
                for se_name in ["sigma1", "sigma2", "sigma3"]:
                    lo, hi = se_est.ci_wald(alpha_hat, X, d, y, which=se_name)
                    sigma_hat = float(getattr(se_est, f"se_{se_name}")(X, d, y))  # asymptotic σ̂_k
                    se_used = float(sigma_hat / np.sqrt(n))                      # finite-sample SE used
                    rows.append(
                        {
                            "rep": rep,
                            "seed": seed,
                            "estimator": est_name,
                            "n": n,
                            "p": p,
                            "tau": tau,
                            "mu": mu,
                            "ci": "wald",
                            "se": se_name,
                            "alpha_true": cfg.alpha_true,
                            "alpha_hat": alpha_hat,
                            "lo": float(lo),
                            "hi": float(hi),
                            "covered": int(lo <= cfg.alpha_true <= hi),
                            "ci_length": float(hi - lo),
                            "support": support,
                            "sigma_hat": sigma_hat,
                            "se_used": se_used,
                        }
                    )

                # Score-based interval (no per-k SE)
                lo_s, hi_s = se_est.ci_score(X, d, y, which=est.__class__.__name__)
                rows.append(
                    {
                        "rep": rep,
                        "seed": seed,
                        "estimator": est_name,
                        "n": n,
                        "p": p,
                        "tau": tau,
                        "mu": mu,
                        "ci": "score",
                        "se": "na",
                        "alpha_true": cfg.alpha_true,
                        "alpha_hat": alpha_hat,
                        "lo": float(lo_s),
                        "hi": float(hi_s),
                        "covered": int(lo_s <= cfg.alpha_true <= hi_s),
                        "ci_length": float(hi_s - lo_s),
                        "support": support,
                        "sigma_hat": float("nan"),
                        "se_used": float("nan"),
                    }
                )
    df = pd.DataFrame.from_records(rows)
    return df

__all__ = ["SimulationConfig", "run_simulation"]