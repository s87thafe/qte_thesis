import numpy as np

from analysis.dgp import sim_dgp
from analysis.double_selection import WeightedDoubleSelection
from analysis.orthogonal_score import OrthogonalScoreEstimator
from analysis.quantile_regression import PenalizedQuantileRegression
from analysis.lasso import AdaptiveLasso
from analysis.density_estimation import ConditionalDensityEstimator


def quick_demo(n: int = 100, p: int = 100, tau: float = 0.5, alpha_true: float = 1.0) -> None:
    """Run a simple demonstration of all components used in the orthogonal score.

    This function simulates a dataset and fits the different estimators that
    compose the orthogonal score approach. It reports the estimated treatment
    effects from each method to illustrate their usage.
    """
    y, d, X, *_ = sim_dgp(n=n, p=p, alpha=alpha_true, seed=0, include_intercept=True)

    y = y.reshape(-1, 1)
    d = d.reshape(-1, 1)

    # Conditional density estimation used for weighting
    # cde = ConditionalDensityEstimator()
    # f_hat = cde.conditional_density_function(X, d, y, tau)

    # Adaptive lasso with post-lasso refit
    # lasso = AdaptiveLasso(post_lasso=True)
    # psi_init         = lasso.penalty_loading_init(f_hat, X, d)
    # theta_post      = lasso.weighted_lasso(X, d, f_hat, psi_init, solver=lasso.solver)
    # psi_updated      = lasso.penalty_loading_updated(f_hat, X, d, theta_post)
    # theta_lasso      = lasso.weighted_lasso(X, d, f_hat, psi_updated, solver=lasso.solver)
    # theta_lasso_post = lasso.fit(X, d, f_hat).theta_post_
    
    # Penalized quantile regression
    # qr = PenalizedQuantileRegression(tau=tau)
    # qr_fit = qr.fit(X, d, y)
    # theta_qr = qr_fit.theta_

    # Weighted double selection estimator
    wds = WeightedDoubleSelection(tau=tau).fit(X, d, y)
    theta_wds = wds.theta_
    wds_selected = wds.n_nonzero_

    # Orthogonal score estimator
    ose = OrthogonalScoreEstimator(tau=tau).fit(X, d, y)
    theta_ose = ose.theta_
    ose_selected = ose.n_nonzero_

    # print(f"f_hat: {f_hat}")
    # print(f"Quantile regression: {theta_qr}")
    print(f"Weighted double selection alpha : {theta_wds[0, 0]}")
    print(f"Weighted double selection non 0 : { wds_selected}")
    print(f"Orthogonal score alpha : {theta_ose[0, 0]}")
    print(f"Orthogonal score non 0 : {ose_selected}")

    # print(f"theta_post        : {theta_post       }")
    # print(f"penalty      : {penalty      }")
    # print(f"weighted_loss  : {weighted_loss}")





if __name__ == "__main__":
    quick_demo()