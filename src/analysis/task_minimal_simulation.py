from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from .config import BLD_data, BLD_figures
from .monte_carlo import SimulationConfig, run_simulation

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def task_run_monte_carlo(
    produces: Path = BLD_data / "monte_carlo.csv",
) -> None:
    """Run the Monte Carlo simulation and store results as a CSV file."""

    cfg = SimulationConfig(
        n_vals=[50],
        p_vals=[50],
        taus=[0.5],
        mu_vals=[1.0],
        n_sim=5,
        seed=0,
    )

    results = run_simulation(cfg)
    produces.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(produces, index=False)


def task_plot_monte_carlo(
    depends: Path = BLD_data / "monte_carlo.csv",
    produces: Path = BLD_figures / "monte_carlo.png",
) -> None:
    """Aggregate from the replicate-level CSV and plot requested summaries."""
    df = pd.read_csv(depends)

    # Aggregate over replications per scenario
    group_keys = ["estimator", "ci", "se", "n", "p", "tau", "mu"]
    agg = (
        df.groupby(group_keys, as_index=False)
          .agg(
              alpha_true=("alpha_true", "first"),
              mean_alpha=("alpha_hat", "mean"),
              std_alpha=("alpha_hat", lambda x: x.std(ddof=1)),
              coverage=("covered", "mean"),
              mean_ci_length=("ci_length", "mean"),
          )
    )
    agg["relative_bias"] = (agg["mean_alpha"] - agg["alpha_true"]) / agg["alpha_true"]
    # Guard: if alpha_true == 0, relative_sd would divide by 0; fall back to absolute sd
    denom = agg["alpha_true"].abs().replace(0, pd.NA)
    agg["relative_sd"] = agg["std_alpha"] / denom

    # Keep a simple slice (as in your example)
    subset = agg[
        (agg["estimator"] == "OS")
        & (agg["ci"] == "wald")
        & (agg["se"] == "sigma1")
        & (agg["mu"] == 0.0)
    ].copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Relative bias vs n",
            "Relative SD vs n",
            "Coverage vs n (95% nominal)",
            "CI length vs coverage"
        )
    )

    # Lines by p for bias, rel SD, coverage
    for p_val, dfi in subset.groupby("p"):
        dfi = dfi.sort_values("n")
        fig.add_trace(
            go.Scatter(x=dfi["n"], y=dfi["relative_bias"],
                       mode="lines+markers", name=f"p={p_val} (bias)"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dfi["n"], y=dfi["relative_sd"],
                       mode="lines+markers", name=f"p={p_val} (rel sd)"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dfi["n"], y=dfi["coverage"],
                       mode="lines+markers", name=f"p={p_val} (coverage)"),
            row=2, col=1
        )

    # 95% nominal line
    n_vals = sorted(subset["n"].unique())
    if len(n_vals) > 0:
        fig.add_trace(
            go.Scatter(x=n_vals, y=[0.95] * len(n_vals),
                       mode="lines", name="95% nominal", line=dict(dash="dash")),
            row=2, col=1
        )

    # CI length vs coverage scatter
    for p_val, dfi in subset.groupby("p"):
        fig.add_trace(
            go.Scatter(x=dfi["mean_ci_length"], y=dfi["coverage"],
                       mode="markers", name=f"p={p_val} (len vs cov)"),
            row=2, col=2
        )

    # Axes titles
    fig.update_xaxes(title_text="n", row=1, col=1)
    fig.update_yaxes(title_text="Relative bias", row=1, col=1)
    fig.update_xaxes(title_text="n", row=1, col=2)
    fig.update_yaxes(title_text="Relative SD", row=1, col=2)
    fig.update_xaxes(title_text="n", row=2, col=1)
    fig.update_yaxes(title_text="Coverage", row=2, col=1)
    fig.update_xaxes(title_text="Mean CI length", row=2, col=2)
    fig.update_yaxes(title_text="Coverage", row=2, col=2)

    fig.update_layout(
        title="Monte Carlo summary (OS, Wald σ̂1, μ=0)",
        legend_title_text="Series"
    )

    produces.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(produces)




__all__ = ["task_run_monte_carlo", "task_plot_monte_carlo"]