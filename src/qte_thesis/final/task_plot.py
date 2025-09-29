from __future__ import annotations

from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import norm

from qte_thesis.config import BLD_data, BLD_figures


def task_plot_bias_boxplots(
    depends: Path = BLD_data / "monte_carlo_merged_v2.csv",
    produces: Path = BLD_figures / "bias_boxplots" / "box_validation.txt",
) -> None:
    """Create bias-vs-n boxplots for each parameter combination."""

    df = pd.read_csv(depends)

    produces.parent.mkdir(parents=True, exist_ok=True)

    group_cols = ["estimator", "p", "mu", "tau"]
    generated_filenames = []
    x_order = [100, 200, 300, 500, 700, 1000]
    for key, dfi in df.groupby(group_cols):

        estimator, p_val, mu_val, tau_val = key
        mu_val = int(mu_val)
        alpha_true = dfi["alpha_true"].iloc[0]

        fig = px.box(dfi, x="n", y="alpha_hat")
        fig.add_hline(
            y=1,
            line_dash="dash",
            annotation_text=f"α={alpha_true}",
            annotation_position="top left",
        )
        fig.update_layout(
        title=(
            f"Boxplot of α̂ and α, by sample size n — {estimator}; p={p_val}, μ={mu_val}, τ={tau_val}"
        ),
            xaxis_title="n",
            yaxis_title="α̂ and α",
            yaxis_range=[-4,6],
            xaxis=dict(tickmode="array", tickvals=x_order, ticktext=[str(v) for v in x_order]),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

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
    depends: Path = BLD_data / "monte_carlo_merged_v2.csv",
    produces: Path = BLD_figures / "mse_plots" / "mse_validation.txt",
) -> None:
    """Save PNGs to output_dir with filenames mirroring the bias-boxplot naming scheme."""
    df = pd.read_csv(depends)

    df["squared_error"] = (df["alpha_hat"] - df["alpha_true"]) ** 2

    omit_outlier = True
    if omit_outlier:
        df = df.dropna(subset=["alpha_hat", "alpha_true"])
        df = df[(abs(df["squared_error"]) <= 100)].copy()

    mse_df = (
        df.groupby(["estimator", "n", "p", "mu", "tau"], dropna=False)["squared_error"]
        .mean()
        .reset_index(name="mse")
        .sort_values(["p", "mu", "tau", "estimator", "n"])
        .reset_index(drop=True)
    )
    # mse_df["rmse"] = np.sqrt(mse_df["mse"])

    produces.parent.mkdir(parents=True, exist_ok=True)

    group_cols = ["p", "mu", "tau"]
    generated_filenames = []
    x_order = [100, 200, 300, 500, 700, 1000]
    for key, dfi in mse_df.groupby(group_cols, dropna=False):
        p_val, mu_val, tau_val = key
        mu_val = int(mu_val)
        fig = px.line(
            dfi.sort_values(["n", "estimator"]),
            x="n",
            y="mse",
            color="estimator",
            markers=True,
        )
        fig.update_yaxes(
            type="log",
            range=[np.log10(0.001), np.log10(5)],
            title_text="MSE of α̂",
        )
        fig.update_xaxes(title_text="Sample size (n)")
        fig.update_layout(
            title=f"MSE vs n (p={p_val}, μ={mu_val}, τ={tau_val})",
            legend_title_text="Estimator",
            xaxis=dict(tickmode="array", tickvals=x_order, ticktext=[str(v) for v in x_order]),
        )
        
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        file_name = f"mse_vs_n_estimator-all_p-{p_val}_mu-{mu_val}_tau-{tau_val}.png"
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    found = len(actual_pngs)
    if expected == found:
        produces.write_text("All MSE plots generated")

def task_plot_coverage_lines(
    depends: Path = BLD_data / "monte_carlo_merged_v2.csv",
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

    ci_order = (
        df["ci_version"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
    )

    palette = px.colors.qualitative.D3
    ci_color_map = {ci: palette[i % len(palette)] for i, ci in enumerate(ci_order)}

    omit_outlier = True
    if omit_outlier:
        df = df.dropna(subset=["lo", "hi"])

    cov_ver = (
        df.groupby(["estimator", "p", "mu", "tau", "n", "ci_version"], dropna=False)["covered"]
        .mean()
        .reset_index(name="coverage")
    )

    group_cols = ["estimator", "p", "mu", "tau"]
    generated_filenames = []
    x_order = [100, 200, 300, 500, 700, 1000]
    for (estimator, p_val, mu_val, tau_val), dfi in cov_ver.groupby(group_cols, sort=False):
        mu_val = int(mu_val)
        dfi = dfi.sort_values("n")

        fig = px.line(
            dfi.sort_values("n"),
            x="n",
            y="coverage",
            color="ci_version",
            markers=True,
            labels={"n": "n", "coverage": "Coverage probability", "ci_version": "CI"},
            color_discrete_map=ci_color_map,
            category_orders={"ci_version": list(ci_order)},
        )

        fig.add_hline(
            y=0.95,
            line_dash="dash",
            annotation_text="target=0.95",
            annotation_position="top left",
        )
        fig.update_yaxes(range=[-0.1, 1.1])
        fig.update_layout(
            title=(
                "Coverage vs n "
                f"(estimator={estimator}, p={p_val}, μ={mu_val}, τ={tau_val})"
            ),
            xaxis_title="n",
            yaxis_title="Coverage probability",
            legend_title_text="CI",
            xaxis=dict(tickmode="array", tickvals=x_order, ticktext=[str(v) for v in x_order]),
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
    depends: Path = BLD_data / "monte_carlo_merged_v2.csv",
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
    x_order = [100, 200, 300, 500, 700, 1000]
    for (estimator, p_val, mu_val, tau_val), dfi in df.groupby(group_cols):
        alpha_true = float(dfi["alpha_true"].iloc[0])
        # Summary of CI lengths by CI version and n
        ci_summ = (
            dfi.groupby(["ci_version", "n"], as_index=False)
               .agg(mean_ci_length=("ci_length", "mean"))
               .sort_values(["ci_version", "n"])
        )
        ci_summ = ci_summ[ci_summ["ci_version"]!="wald-sigma2"]
        ahat = (
            dfi.groupby("n", as_index=False)
               .agg(mean_alpha_hat=("alpha_hat", "mean"))
               .sort_values("n")
        )
        ci_summ = ci_summ.merge(ahat, on="n", how="left")
        ci_summ["ci_half"] = ci_summ["mean_ci_length"] / 2.0

        ci_versions = ci_summ["ci_version"].unique()
        offsets = np.linspace(-0.18, 0.18, len(ci_versions))  
        offset_map = dict(zip(ci_versions, offsets))
        ci_summ["x_pos"] = ci_summ["n"] + ci_summ["ci_version"].map(offset_map)

        # Scatter with error bars centered at mean_alpha_hat
        fig = px.scatter(
            ci_summ,
            x="x_pos",
            y="mean_alpha_hat",
            error_y="ci_half",
            color="ci_version",
            title=(
                "Avg CI length vs n "
                f"(estimator={estimator}, p={p_val}, μ={mu_val}, τ={tau_val})"
            ),
            labels={"x_pos": "n", "mean_alpha_hat": "α̂", "ci_version": "CI version"},
        )
        fig_alpha = px.line(ahat, x="n", y="mean_alpha_hat", markers=True)
        fig_alpha.update_traces(name="mean α̂", showlegend=True)
        for tr in fig_alpha.data:
            fig.add_trace(tr)

        if estimator == "OS":
            fig.update_layout(
                xaxis_title="n",
                yaxis_title="α",
                yaxis_range=[0, 2],
                legend_title="Series",
            )
        else:
            fig.update_layout(
                xaxis_title="n",
                yaxis_title="α",
                yaxis_range=[-3, 3],
                legend_title="Series",
                xaxis=dict(tickmode="array", tickvals=x_order, ticktext=[str(v) for v in x_order]),
            )

        file_name = (
            f"ci_len_vs_n_estimator-{estimator}_p-{p_val}_mu-{mu_val}_tau-{tau_val}.png"
        )
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    if expected == len(actual_pngs):
        produces.write_text("All CI length plots generated")