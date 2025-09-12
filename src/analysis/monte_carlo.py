"""Monte Carlo simulation framework for treatment effect estimators.

This module provides a convenience function :func:`run_simulation` that
compares the orthogonal score (OS) and weighted double selection (DS)
estimators across a collection of design parameters.  For each scenario the
function repeatedly simulates data, fits the estimators and evaluates a range
of performance metrics such as relative bias, standard deviation, confidence
interval coverage and average support size.

The simulation varies

* estimator: Orthogonal Score vs Weighted Double Selection;
* standard errors: ``sigma1``, ``sigma2`` and ``sigma3`` from
  :class:`analysis.standard_error.StandardErrorEstimator`;
* confidence intervals: Wald (using the three standard errors) and the score
  based interval;
* data generating process: homoscedastic (``mu=0``) and heteroscedastic
  (``mu>0``);
* quantile levels ``tau`` as well as sample sizes ``n`` and number of
  covariates ``p``.

The function returns a :class:`pandas.DataFrame` summarising the Monte Carlo
results for every combination of the above dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Dict, Any

import numpy as np
import pandas as pd

from .dgp import sim_dgp
from .double_selection import WeightedDoubleSelection
from .orthogonal_score import OrthogonalScoreEstimator
from .standard_error import StandardErrorEstimator


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


# ---------------------------------------------------------------------------
# Core simulation routine
# ---------------------------------------------------------------------------

def run_simulation(cfg: SimulationConfig) -> pd.DataFrame:
    """Run the Monte Carlo study and return a dataframe of summary metrics.

    The resulting dataframe contains one row for each combination of
    estimator, standard error, confidence interval type and design parameters.
    Columns include relative bias, standard deviation, coverage probability of
    the nominal 95\% interval, average interval length and average selected
    support size.
    """

    rng = np.random.default_rng(cfg.seed)

    records: Dict[tuple, List[Dict[str, Any]]] = {}

    # Iterate over all design parameters
    for n, p, tau, mu in product(cfg.n_vals, cfg.p_vals, cfg.taus, cfg.mu_vals):
        for sim in range(cfg.n_sim):
            # Draw a fresh seed for every replication
            seed = int(rng.integers(1e9))
            y, d, X, *_ = sim_dgp(n=n, p=p, mu=mu, alpha=cfg.alpha_true, seed=seed)
            y = y.reshape(-1, 1)
            d = d.reshape(-1, 1)

            for est_name, Est in [
                ("DS", WeightedDoubleSelection),
                ("OS", OrthogonalScoreEstimator),
            ]:
                estimator = Est(tau=tau).fit(X, d, y)
                alpha_hat = float(estimator.theta_[0, 0])
                support = int(estimator.n_nonzero_)

                se_est = StandardErrorEstimator(tau=tau)

                # Wald intervals for each of the three standard errors
                se_funcs = ["sigma1", "sigma2", "sigma3"]
                for se_name in se_funcs:
                    lo, hi = se_est.ci_wald(alpha_hat, X, d, y, which=se_name)
                    key = (est_name, n, p, tau, mu, "wald", se_name)
                    records.setdefault(key, []).append(
                        {
                            "alpha_hat": alpha_hat,
                            "lo": lo,
                            "hi": hi,
                            "support": support,
                        }
                    )

                # Score based interval
                lo, hi = se_est.ci_score(X, d, y, which=estimator.__class__.__name__)
                key = (est_name, n, p, tau, mu, "score", "na")
                records.setdefault(key, []).append(
                    {
                        "alpha_hat": alpha_hat,
                        "lo": lo,
                        "hi": hi,
                        "support": support,
                    }
                )

    # Compute summary statistics for each scenario
    rows = []
    for key, vals in records.items():
        est_name, n, p, tau, mu, ci, se_name = key
        alpha_hats = np.array([v["alpha_hat"] for v in vals])
        supports = np.array([v["support"] for v in vals])
        covers = np.array([(cfg.alpha_true >= v["lo"]) and (cfg.alpha_true <= v["hi"]) for v in vals])
        lengths = np.array([v["hi"] - v["lo"] for v in vals])

        mean_alpha = alpha_hats.mean()
        rel_bias = (mean_alpha - cfg.alpha_true) / cfg.alpha_true
        std = alpha_hats.std(ddof=1)
        coverage = covers.mean()
        mean_length = lengths.mean()
        mean_support = supports.mean()

        rows.append(
            {
                "estimator": est_name,
                "n": n,
                "p": p,
                "tau": tau,
                "mu": mu,
                "ci": ci,
                "se": se_name,
                "relative_bias": rel_bias,
                "std": std,
                "coverage": coverage,
                "ci_length": mean_length,
                "support_size": mean_support,
            }
        )

    return pd.DataFrame(rows)


__all__ = ["SimulationConfig", "run_simulation"]