# simulation_5.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from qte_thesis.config import BLD_data
from final.monte_carlo import SimulationConfig, run_simulation

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-id", type=int, required=True, help="0..N-1")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()

def run_one(sim_id: int, produces: Path) -> None:
    # Base config; we override n_sim and seed below
    base = SimulationConfig(
        n_vals=[100, 200, 500, 1000],
        p_vals=[1000],
        taus=[0.05, 0.5, 0.95],
        mu_vals=[0.0, 1.0],
        n_sim=1,                
        seed=1 + sim_id * 100_000,  
    )

    df = run_simulation(base)
    df.insert(0, "sim_id", sim_id)
    produces.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(produces, index=False)

if __name__ == "__main__":
    args = parse_args()
    out = args.out or (BLD_data / f"monte_carlo_part_{args.sim_id:03d}.csv")
    run_one(args.sim_id, out)
