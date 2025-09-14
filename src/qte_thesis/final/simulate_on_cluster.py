from __future__ import annotations

from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Any, List, Sequence

import numpy as np
import pandas as pd

from qte_thesis.config import BLD_data
from .monte_carlo import SimulationConfig
from .dgp import sim_dgp
from qte_thesis.analysis.double_selection import WeightedDoubleSelection
from qte_thesis.analysis.orthogonal_score import OrthogonalScoreEstimator
from qte_thesis.analysis.standard_error import StandardErrorEstimator

def _single_rep(task: Sequence[Any]) -> List[Dict[str, Any]]:
    """Run a single Monte Carlo replication.

    Parameters
    ----------
    task
        Tuple ``(n, p, tau, mu, rep, seed, alpha_true)`` describing one
        simulation setup.
    """
    n, p, tau, mu, rep, seed, alpha_true = task
    y, d, X, *_ = sim_dgp(n=n, p=p, mu=mu, alpha=alpha_true, seed=seed)
    y = y.reshape(-1, 1)
    d = d.reshape(-1, 1)

    rows: List[Dict[str, Any]] = []
    for est_name, Est in [
        ("DS", WeightedDoubleSelection),
        ("OS", OrthogonalScoreEstimator),
    ]:
        est = Est(tau=tau).fit(X, d, y)
        alpha_hat = float(est.theta_[0, 0])
        support = int(est.n_nonzero_)

        se_est = StandardErrorEstimator(tau=tau)

        # Wald intervals for each of the three SE estimators
        for se_name in ["sigma1", "sigma2", "sigma3"]:
            lo, hi = se_est.ci_wald(alpha_hat, X, d, y, which=se_name)
            sigma_hat = float(getattr(se_est, f"se_{se_name}")(X, d, y))
            se_used = float(sigma_hat / np.sqrt(n))
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
                    "alpha_true": alpha_true,
                    "alpha_hat": alpha_hat,
                    "lo": float(lo),
                    "hi": float(hi),
                    "covered": int(lo <= alpha_true <= hi),
                    "ci_length": float(hi - lo),
                    "support": support,
                    "sigma_hat": sigma_hat,
                    "se_used": se_used,
                }
            )

        # Score-based interval
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
                "alpha_true": alpha_true,
                "alpha_hat": alpha_hat,
                "lo": float(lo_s),
                "hi": float(hi_s),
                "covered": int(lo_s <= alpha_true <= hi_s),
                "ci_length": float(hi_s - lo_s),
                "support": support,
                "sigma_hat": float("nan"),
                "se_used": float("nan"),
            }
        )

    return rows


def run_simulation_parallel(
    cfg: SimulationConfig, n_jobs: int | None = None
) -> pd.DataFrame:
    """Run the Monte Carlo experiment in parallel.

    Parameters
    ----------
    cfg
        Configuration for the simulation.
    n_jobs
        Number of worker processes.  Defaults to all available cores.
    """
    rng = np.random.default_rng(cfg.seed)

    tasks: List[Sequence[Any]] = []
    for n, p, tau, mu in product(cfg.n_vals, cfg.p_vals, cfg.taus, cfg.mu_vals):
        for rep in range(cfg.n_sim):
            seed = int(rng.integers(123))
            tasks.append((n, p, tau, mu, rep, seed, cfg.alpha_true))

    if not n_jobs or n_jobs <= 0:
        n_jobs = cpu_count()

    with Pool(processes=n_jobs) as pool:
        results = pool.map(_single_rep, tasks)

    rows = [row for sub in results for row in sub]
    return pd.DataFrame.from_records(rows)


def task_run_monte_carlo_parallel(
    produces: Path = BLD_data / "monte_carlo_parallel.csv",
    n_jobs: int | None = None,
) -> None:
    """Run the Monte Carlo simulation in parallel and store results."""
    cfg = SimulationConfig(
        n_vals=[100, 200, 500, 1000],
        p_vals=[1000],
        taus=[0.05, 0.5, 0.95],
        mu_vals=[0.0, 1.0],
        n_sim=500,
        seed=0,
    )

    results = run_simulation_parallel(cfg, n_jobs=n_jobs)
    produces.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(produces, index=False)


__all__ = [
    "run_simulation_parallel",
    "task_run_monte_carlo_parallel",
]