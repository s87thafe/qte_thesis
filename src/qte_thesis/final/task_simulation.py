from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from qte_thesis.config import BLD_data, BLD_figures
from .monte_carlo import SimulationConfig, run_simulation

import plotly.graph_objects as go



def task_run_monte_carlo(
    produces: Path = BLD_data / "monte_carlo.csv",
) -> None:
    """Run the Monte Carlo simulation and store results as a CSV file."""
    cfg = SimulationConfig(
        n_vals=[100, 300, 1000],   
        p_vals=[500, 1000],      
        taus=[0.05, 0.25, 0.5, 0.75, 0.95],
        mu_vals=[0.0, 1.0],
        n_sim=1000,
        seed=0,
    )

    results = run_simulation(cfg)
    produces.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(produces, index=False)


__all__ = ["task_run_monte_carlo", "task_plot_bias_boxplots"]