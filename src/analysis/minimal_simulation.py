"""Minimal example running the Monte Carlo simulation once per setting."""
from __future__ import annotations

from analysis import SimulationConfig, run_simulation


def main() -> None:
    """Run the Monte Carlo simulation for a single configuration.

    This example exercises both estimators (OS and DS), all three standard
    error formulas and both confidence interval types on a tiny design. Each
    scenario is run only once to keep the runtime minimal.
    """

    cfg = SimulationConfig(
        n_vals=[50],
        p_vals=[10],
        taus=[0.5],
        mu_vals=[0.0, 1.0],
        n_sim=1,
        seed=0,
    )

    results = run_simulation(cfg)
    print(results)


if __name__ == "__main__":
    main()