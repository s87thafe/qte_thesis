from __future__ import annotations

from pathlib import Path
import plotly.express as px
import pandas as pd

from qte_thesis.config import BLD_data, BLD_figures


def task_plot_bias_boxplots(
    depends: Path = BLD_data / "monte_carlo.csv",
    produces: Path = BLD_figures / "bias_boxplots" / "validation.txt",
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
        with open(produces, "x") as f:
            f.write("All boxplots generated")



def task_plot_mse(   
    depends: Path = BLD_data / "monte_carlo.csv",
    produces: Path = BLD_figures / "mse_plots" / "validation.txt",
) -> None:
    """
    For each (p, mu, tau) combination, make one Matplotlib line chart:
    - x: n
    - y: MSE (log scale)
    - one line per estimator
    Save PNGs to output_dir with filenames mirroring the bias-boxplot naming scheme.
    """
    df = pd.read_csv(depends)

    df["squared_error"] = (df["alpha_hat"] - df["alpha_true"]) ** 2

    grouped = (
        df.groupby(["estimator", "n", "p", "tau", "mu"], dropna=False)["squared_error"]
        .agg(mse="mean", reps="size")
        .reset_index()
    )
    grouped = grouped.sort_values(["p", "mu", "tau", "estimator", "n"]).reset_index(drop=True)

    # Unique combos
    combos = (
        df[["p", "mu", "tau"]]
        .drop_duplicates()
        .sort_values(["p", "mu", "tau"])
        .to_dict(orient="records")
    )
    generated_filenames = []
    for combo in combos:
        p_val, mu, tau = combo["p"], combo["mu"], combo["tau"]
        sub = df[(df["p"] == p_val) & (df["mu"] == mu) & (df["tau"] == tau)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for est in sorted(sub["estimator"].unique()):
            se = sub[sub["estimator"] == est].sort_values("n")
            ax.plot(se["n"].values, se["mse"].values, marker="o", label=str(est))

        ax.set_yscale("log")  # log scale for MSE
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("MSE of α̂")
        ax.set_title(f"MSE vs n — p={p_val} | μ={mu} | τ={tau}")
        ax.legend(title="Estimator")
        ax.grid(True, which="both", axis="both")

        file_name = f"mse_vs_n_estimator-all_p-{p_val}_mu-{mu}_tau-{tau}.png"
        fig.write_image(produces.parent / file_name)
        generated_filenames.append(file_name)

    expected = len(generated_filenames)
    actual_pngs = sorted(p.name for p in produces.parent.glob("*.png"))
    found = len(actual_pngs)
    if expected == found:
        with open(produces, "x") as f:
            f.write("All boxplots generated")

