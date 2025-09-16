from __future__ import annotations

from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import norm

from qte_thesis.config import BLD_data, BLD_figures


def task_plot_bias_boxplots(
    depends: Path = BLD_data / "monte_carlo_merged.csv",
    produces: Path = BLD_figures / "bias_boxplots" / "box_validation.txt",
) -> None:
    """Create bias-vs-n boxplots for each parameter combination.

    For each combination of estimator, p, mu, and tau, a boxplot of the bias
    (α̂ − α) across Monte Carlo replications is saved. A dashed horizontal line
    at zero indicates the true value (no bias).
    """

    df = pd.read_csv(depends)
    df["bias"] = df["alpha_hat"] - df["alpha_true"]

    produces.parent.mkdir(parents=True, exist_ok=True)

    group_cols = ["estimator", "p", "mu", "tau"]
    generated_filenames = []
    for key, dfi in df.groupby(group_cols):

        estimator, p_val, mu_val, tau_val = key
        alpha_true = dfi["alpha_true"].iloc[0]

        fig = px.box(dfi, x="n", y="bias")
        fig.add_hline(
            y=0,
            line_dash="dash",
            annotation_text=f"α={alpha_true}",
            annotation_position="top left",
        )
        fig.update_layout(
            title=(
                "Bias vs n "
                f"(estimator={estimator}, p={p_val}, μ={mu_val}, τ={tau_val})"
            ),
            xaxis_title="n",
            yaxis_title="Bias (α̂ − α)",
        )

        file_name = (
            f"bias_vs_n_estimator-{estimator}_p-{p_val}_mu-{mu_val}_tau-{tau_val}.png"
        )
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    found = len(actual_pngs)
    if expected == found:
        produces.write_text("All boxplots generated")



def task_plot_mse(   
    depends: Path = BLD_data / "monte_carlo_merged.csv",
    produces: Path = BLD_figures / "mse_plots" / "mse_validation.txt",
) -> None:
    """
    For each (p, mu, tau) combination, create a Plotly line chart:
    - x: n
    - y: MSE (log scale)
    - one line per estimator
    Save PNGs to output_dir with filenames mirroring the bias-boxplot naming scheme.
    """
    df = pd.read_csv(depends)
    df["squared_error"] = (df["alpha_hat"] - df["alpha_true"]) ** 2

    # Aggregate MSE per (estimator, n, p, mu, tau)
    mse_df = (
        df.groupby(["estimator", "n", "p", "mu", "tau"], dropna=False)["squared_error"]
        .mean()
        .reset_index(name="mse")
        .sort_values(["p", "mu", "tau", "estimator", "n"])
        .reset_index(drop=True)
    )

    produces.parent.mkdir(parents=True, exist_ok=True)

    group_cols = ["p", "mu", "tau"]
    generated_filenames = []
    for key, dfi in mse_df.groupby(group_cols, dropna=False):
        p_val, mu_val, tau_val = key

        fig = px.line(
            dfi.sort_values(["n", "estimator"]),
            x="n",
            y="mse",
            color="estimator",
            markers=True,
        )
        fig.update_yaxes(type="log", title_text="MSE of α̂")
        fig.update_xaxes(title_text="Sample size (n)")
        fig.update_layout(
            title=f"MSE vs n (p={p_val}, μ={mu_val}, τ={tau_val})",
            legend_title_text="Estimator",
        )

        file_name = f"mse_vs_n_estimator-all_p-{p_val}_mu-{mu_val}_tau-{tau_val}.png"
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    found = len(actual_pngs)
    if expected == found:
        produces.write_text("All MSE plots generated")

def task_plot_coverage_lines(
    depends: Path = BLD_data / "monte_carlo_merged.csv",
    produces: Path = BLD_figures / "coverage_lines" / "coverage_validation.txt",
) -> None:
    """Create coverage-vs-n line plots using precomputed 'covered' and 'ci'."""
    df = pd.read_csv(depends)

    produces.parent.mkdir(parents=True, exist_ok=True)

    df["ci_version"] = np.where(
        df["ci"].astype(str).str.lower().eq("wald"),
        "wald-" + df["se"].astype(str),
        df["ci"].astype(str),
    )

    cov_ver = (
        df.groupby(["estimator", "p", "mu", "tau", "n", "ci_version"], dropna=False)["covered"]
        .mean()
        .reset_index(name="coverage")
    )

    group_cols = ["estimator", "p", "mu", "tau"]
    generated_filenames = []

    for (estimator, p_val, mu_val, tau_val), dfi in cov_ver.groupby(group_cols, sort=False):
        dfi = dfi.sort_values("n")

        fig = px.line(
            dfi,
            x="n",
            y="coverage",
            color="ci_version",
            markers=True,
            labels={"n": "n", "coverage": "Coverage probability", "ci_version": "CI"},
        )

        fig.add_hline(
            y=0.95,
            line_dash="dash",
            annotation_text="target=0.95",
            annotation_position="top left",
        )
        fig.update_yaxes(range=[0.0, 1.0])
        fig.update_layout(
            title=(
                "Coverage vs n "
                f"(estimator={estimator}, p={p_val}, μ={mu_val}, τ={tau_val})"
            ),
            xaxis_title="n",
            yaxis_title="Coverage probability",
            legend_title_text="CI",
            margin=dict(l=60, r=20, t=60, b=50),
        )

        file_name = (
            f"coverage_by_n_estimator-{estimator}_p-{p_val}_mu-{mu_val}_tau-{tau_val}.png"
        )
        fig.write_image(produces.parent / file_name, scale=2)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    if expected == len(actual_pngs):
        produces.write_text("All coverage plots generated")


def task_plot_ci_length(
    depends: Path = BLD_data / "monte_carlo_merged.csv",
    produces: Path = BLD_figures / "ci_length_plots" / "ci_validation.txt",
) -> None:
    """Create CI-length-vs-n plots for each parameter combination."""
    df = pd.read_csv(depends)

    # Build CI label: 'score' or 'wald-<se>'
    df["ci_version"] = df.apply(
        lambda r: f"wald-{r['se']}" if str(r["ci"]).lower() == "wald" else str(r["ci"]),
        axis=1,
    )

    produces.parent.mkdir(parents=True, exist_ok=True)

    group_cols = ["estimator", "p", "mu", "tau"]
    generated_filenames = []

    for (estimator, p_val, mu_val, tau_val), dfi in df.groupby(group_cols):
        alpha_true = float(dfi["alpha_true"].iloc[0])
        ci_summ = (
            dfi.groupby(["ci_version", "n"], as_index=False)
               .agg(mean_ci_length=("ci_length", "mean"))
               .sort_values(["ci_version", "n"])
        )
        ci_summ["alpha_center"] = alpha_true
        ci_summ["ci_half"] = ci_summ["mean_ci_length"] / 2.0
        ahat = (
            dfi.groupby("n", as_index=False)
               .agg(mean_alpha_hat=("alpha_hat", "mean"))
               .sort_values("n")
        )
        fig = px.scatter(
            ci_summ,
            x="n",
            y="alpha_center",
            error_y="ci_half",
            color="ci_version",
            title=(
                "Avg CI length vs n "
                f"(estimator={estimator}, p={p_val}, μ={mu_val}, τ={tau_val})"
            ),
            labels={"n": "n", "alpha_center": "α", "ci_version": "CI version"},
        )
        fig_alpha = px.line(ahat, x="n", y="mean_alpha_hat", markers=True)
        fig_alpha.update_traces(name="mean α̂", showlegend=True)
        for tr in fig_alpha.data:
            fig.add_trace(tr)
        fig.add_hline(
            y=alpha_true,
            line_dash="dash",
            annotation_text=f"α={alpha_true}",
            annotation_position="top left",
        )
        fig.update_layout(xaxis_title="n", yaxis_title="α", legend_title="Series")

        file_name = (
            f"ci_len_vs_n_estimator-{estimator}_p-{p_val}_mu-{mu_val}_tau-{tau_val}.png"
        )
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    if expected == len(actual_pngs):
        produces.write_text("All CI length plots generated")


def task_plot_bias_boxplots(
    depends: Path = BLD_data / "monte_carlo_merged.csv",
    produces: Path = BLD_figures / "bias_boxplots" / "box_validation.txt",
) -> None:
    """Create QQ plots of studentized √n-errors vs N(0,1) for each parameter combination."""

    df = pd.read_csv(depends)
    df["_t"] = np.sqrt(df["n"].astype(float)) * (
        df["alpha_hat"].astype(float) - df["alpha_true"].astype(float)
    ) / df["sigma_hat"].astype(float)

    produces.parent.mkdir(parents=True, exist_ok=True)

    group_cols = ["estimator", "se", "p", "mu", "tau", "n"]
    generated_filenames = []

    for key, dfi in df.groupby(group_cols):
        estimator, se_name, p_val, mu_val, tau_val, n_val = key

        s = np.sort(dfi["_t"].to_numpy(dtype=float))
        m = s.size
        if m == 0:
            continue
        probs = (np.arange(1, m + 1) - 0.5) / m
        theo = norm.ppf(probs)  # SciPy

        qqdf = pd.DataFrame({"theoretical": theo, "sample": s})
        fig = px.scatter(qqdf, x="theoretical", y="sample")

        lo = float(min(qqdf["theoretical"].min(), qqdf["sample"].min()))
        hi = float(max(qqdf["theoretical"].max(), qqdf["sample"].max()))
        # 45° line without go.Scatter
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi)

        fig.update_layout(
            title=(
                "QQ plot: √n(α̂−α)/σ̂ vs N(0,1) "
                f"(estimator={estimator}, se={se_name}, p={p_val}, μ={mu_val}, τ={tau_val}, n={n_val})"
            ),
            xaxis_title="Theoretical N(0,1) quantiles",
            yaxis_title="Sample quantiles",
            showlegend=False,
        )

        file_name = (
            f"qq_estimator-{estimator}_se-{se_name}_p-{p_val}_mu-{mu_val}_tau-{tau_val}_n-{n_val}.png"
        )
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    found = len(actual_pngs)
    if expected == found:
        produces.write_text("All QQ plots generated")