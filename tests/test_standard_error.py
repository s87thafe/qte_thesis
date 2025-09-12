import numpy as np
from analysis.dgp import sim_dgp
from analysis.double_selection import WeightedDoubleSelection
from analysis.standard_error import StandardErrorEstimator


def simulate_data(seed: int = 123):
    """Generate a small synthetic dataset for testing."""
    y, d, X, _, _, _ = sim_dgp(n=100, p=20, alpha=1.0, seed=seed, include_intercept=False)
    return y.reshape(-1, 1), d.reshape(-1, 1), X


def test_standard_error_estimators_positive():
    """All standard error estimators should return positive finite values."""
    y, d, X = simulate_data()
    see = StandardErrorEstimator(tau=0.5)
    sigma1 = see.se_sigma1(X, d, y)
    sigma2 = see.se_sigma2(X, d, y)
    sigma3 = see.se_sigma3(X, d, y)
    for val in (sigma1, sigma2, sigma3):
        assert np.isfinite(val)
        assert val > 0


def test_confidence_intervals_return_bounds():
    """Wald and score confidence intervals should return valid bounds."""
    y, d, X = simulate_data(seed=321)
    alpha_hat = WeightedDoubleSelection(tau=0.5).fit(X, d, y).theta_[0, 0]
    see = StandardErrorEstimator(tau=0.5)
    for which in ["sigma1", "sigma2", "sigma3"]:
        lo, hi = see.ci_wald(alpha_hat, X, d, y, level=0.95, which=which)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi > lo
    lo1, hi1 = see.ci_score(X, d, y, level=0.95, which="WeightedDoubleSelection")
    lo2, hi2 = see.ci_score(X, d, y, level=0.95, which="OrthogonalScoreEstimator")
    for lo, hi in [(lo1, hi1), (lo2, hi2)]:
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi > lo