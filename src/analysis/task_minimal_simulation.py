from __future__ import annotations

from pathlib import Path

from .monte_carlo import SimulationConfig, run_simulation
from .config import BLD_data

def task_run_monte_carlo(
    produces: Path = BLD_data / "monte_carlo.csv",
) -> None:
    """Run the Monte Carlo simulation and store results as a CSV file."""
    cfg = SimulationConfig(
        n_vals=[50],
        p_vals=[10],
        taus=[0.5],
        mu_vals=[0.0, 1.0],
        n_sim=1,
        seed=0,
    )

    results = run_simulation(cfg)
    produces.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(produces, index=False)