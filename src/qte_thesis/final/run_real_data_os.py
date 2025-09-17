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
from qte_thesis.analysis.quantile_regression import PenalizedQuantileRegression

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, required=True, help="Quantile in (0,1).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for subsampling.")
    p.add_argument(
        "--n-subsample", dest="n_subsample", type=int, default=None,
        help="Subsample size; if None, use full data."
    )
    p.add_argument(
        "--subsample-id", dest="subsample_id", type=int, default=None,
        help="Identifier for the subsample (for logging/output filenames)."
    )
    p.add_argument(
        "--csv-source", type=Path, default=BLD_data / "real_data.csv",
        help="Path to input CSV (defaults to real_data.csv)."
    )
    return p.parse_args()

def run_os(tau: float, csv_source: Path, produces: Path, seed: int | None = None, n_subsample: int | None = None, subsample_id: int | None = None) -> None:

    rng = np.random.default_rng(seed)
    ipumps_data = pd.read_csv(csv_source)

    if n_subsample is not None and n_subsample < len(ipumps_data):
        idx = rng.choice(len(ipumps_data), size=n_subsample, replace=False)
        ipumps_data = ipumps_data.iloc[idx].reset_index(drop=True)

    y = ipumps_data[["HOURLY_WAGE"]].to_numpy(dtype=float)  
    d = ipumps_data[["min_wage"]].to_numpy(dtype=float)     
    Xdf = ipumps_data.drop(columns=["HOURLY_WAGE", "min_wage"])
    X = Xdf.to_numpy(dtype=float)
    n = y.size

    # Test Singularity
    psi = PenalizedQuantileRegression.compute_diagonal_psi(d, X)
    diag = np.diag(psi).astype(float)
    tol = 1e-12
    bad_full = (~np.isfinite(diag)) | (np.abs(diag) <= tol)
    if bad_full[0]:
        raise ValueError("No variation in d in this split.")

    drop_X_idx = [i - 1 for i in np.where(bad_full)[0] if i != 0]
    keep = np.setdiff1d(np.arange(X.shape[1]), drop_X_idx)
    X = X[:, keep]

    eps = 1e-12
    var = X.var(axis=0)
    keep = var > eps
    X = X[:, keep]
    _, uniq_idx = np.unique(X, axis=1, return_index=True)
    X = X[:, np.sort(uniq_idx)]

    X = np.column_stack([np.ones(n), X])

    est = OrthogonalScoreEstimator(tau=tau).fit(X, d, y)
    alpha_hat = float(est.theta_[0, 0])
    support = int(est.n_nonzero_)

    se_est = StandardErrorEstimator(tau=tau)

    rows = []
    rows = []
    for se_name in ["sigma1", "sigma2", "sigma3"]:
        try:
            lo, hi = se_est.ci_wald(alpha_hat, X, d, y, which=se_name)
            sigma_hat = float(getattr(se_est, f"se_{se_name}")(X, d, y))
            se_used = float(sigma_hat / np.sqrt(n))
            rows.append({
                "estimator": "OS",
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
                "subsample_id": subsample_id,
                "n_subsample": len(ipumps_data),
                "seed": seed,
            })
        except Exception as e:
            rows.append({
                "estimator": "OS", "tau": tau, "ci": "wald", "se": se_name,
                "alpha_hat": alpha_hat, "lo": np.nan, "hi": np.nan,
                "ci_length": np.nan, "support": support,
                "sigma_hat": np.nan, "se_used": np.nan,
                "error": f"{type(e).__name__}: {e}"
            })
    produces.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(produces, index=False)


if __name__ == "__main__":
    args = parse_args()
    out = BLD_data / f"real_data_{args.tau}_os_sub{args.subsample_id}.csv"
    run_os(
        tau=args.tau,
        csv_source=args.csv_source,
        produces=out,
        seed=args.seed,
        n_subsample=args.n_subsample,
        subsample_id=args.subsample_id,
    )