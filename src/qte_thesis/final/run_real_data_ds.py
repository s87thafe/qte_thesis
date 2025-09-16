from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from typing import Iterable, List, Dict, Any
import numpy as np
from qte_thesis.config import BLD_data
from qte_thesis.analysis.double_selection import WeightedDoubleSelection
from qte_thesis.analysis.orthogonal_score import OrthogonalScoreEstimator
from qte_thesis.analysis.standard_error import StandardErrorEstimator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, required=True, help="Quantile in (0,1).")
    return p.parse_args()

def run_ds(tau: int, csv_source: Path, produces: Path) -> None:
    rows: List[Dict[str, Any]] = []

    ipumps_data = pd.read_csv(csv_source)
    y = ipumps_data[["HOURLY_WAGE"]].to_numpy(dtype=float)  
    d = ipumps_data[["min_wage"]].to_numpy(dtype=float)     
    X = ipumps_data.drop(columns=["HOURLY_WAGE", "min_wage"]).to_numpy(dtype=float)
    n = y.size

    est = WeightedDoubleSelection(tau=tau).fit(X, d, y)
    alpha_hat = float(est.theta_[0, 0])
    support = int(est.n_nonzero_)

    se_est = StandardErrorEstimator(tau=tau)

    rows: List[Dict[str, Any]] = []
    # Wald intervals for each of the three SEs
    for se_name in ["sigma1", "sigma2", "sigma3"]:
        lo, hi = se_est.ci_wald(alpha_hat, X, d, y, which=se_name)
        sigma_hat = float(getattr(se_est, f"se_{se_name}")(X, d, y))  # asymptotic σ̂_k
        se_used = float(sigma_hat / np.sqrt(n))                      # finite-sample SE used
        rows.append(
            {
                "estimator": "DS",
                "tau": tau,
                "ci": "wald",
                "se": se_name,
                "alpha_hat": alpha_hat,
                "lo": float(lo),
                "hi": float(hi),
                "ci_length": float(hi - lo),
                "support": support,
                "sigma_hat": sigma_hat,
                "se_used": se_used,
            }
        )
    produces.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(rows)
    df.to_csv(produces, index=False)


if __name__ == "__main__":
    args = parse_args()
    run_ds(
        tau=args.tau,
        csv_source=BLD_data / "real_data.csv",
        produces=BLD_data / f"real_data_{args.tau}_ds.csv",
    )