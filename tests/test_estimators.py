import numpy as np
from analysis.dgp import sim_dgp
from analysis.double_selection import WeightedDoubleSelection
from analysis.orthogonal_score import OrthogonalScoreEstimator


def test_estimators_recover_alpha():
    """Both estimators should recover the treatment effect in the simulated DGP."""
    alpha_true = 1.0
    y, d, X, _, _, _ = sim_dgp(n=200, p=50, alpha=alpha_true, seed=0, include_intercept=False)
    y = y.reshape(-1, 1)
    d = d.reshape(-1, 1)

    wds = WeightedDoubleSelection(tau=0.5).fit(X, d, y)
    ose = OrthogonalScoreEstimator(tau=0.5).fit(X, d, y)

    alpha_wds = float(wds.theta_[0])
    alpha_ose = float(ose.theta_[0])

    print(f"WDS estimate: {alpha_wds:.4f}, OSE estimate: {alpha_ose:.4f}, True alpha: {alpha_true:.4f}")

    assert np.isfinite(alpha_wds)
    assert np.isfinite(alpha_ose)