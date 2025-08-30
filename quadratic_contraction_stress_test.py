#!/usr/bin/env python3
"""
Quadratic-contraction stress tests for the RG recurrence:

  Ideal:      eta_{k+1} = A * eta_k^2
  Var-A:      A_k in [A*(1-rho), A*(1+rho)]
  Leak (bad): eta_{k+1} = A * eta_k^2 + eps * eta_k   # simulates an n=1 contamination

Outputs:
  - CSV summaries (all scenarios + leak scenarios)
  - Two log-scale plots:
      1) Ideal contraction trajectories across several A
      2) Ideal vs linear-leak comparison at A=50

Author: Keefe Reeves
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)

@dataclass
class ScenarioConfig:
    name: str
    A: float
    alpha: float     # eta0 = alpha / A (stress near KP boundary)
    steps: int = 60
    vary_A: bool = False
    rho: float = 0.0
    eps: float = 0.0  # linear leak

def run_sequence(cfg: ScenarioConfig) -> Tuple[List[float], List[float], List[float], Dict]:
    """
    Run the recurrence under the specified scenario.
    Returns:
      - seq: the sequence [eta_0, eta_1, ..., eta_T]
      - Ak_list: per-step A_k used
      - effA: empirical effective A_k := eta_{k+1} / eta_k^2 (when defined)
      - result dict with summary metrics
    """
    tiny = 1e-320          # stop when below this (avoid underflow)
    denom_min = 1e-300     # guard for effA division

    eta0 = cfg.alpha / cfg.A
    seq: List[float] = [eta0]
    effA: List[float] = []
    Ak_list: List[float] = []
    eta = eta0

    for _k in range(cfg.steps):
        if cfg.vary_A:
            delta = random.uniform(-cfg.rho, cfg.rho)
            Ak = cfg.A * (1.0 + delta)
        else:
            Ak = cfg.A
        Ak_list.append(Ak)

        next_eta = Ak * (eta ** 2) + cfg.eps * eta
        seq.append(next_eta)

        denom = eta * eta
        if denom > denom_min:
            effA.append(next_eta / denom)
        else:
            effA.append(float('nan'))

        eta = next_eta
        if eta < tiny:
            break
        if not math.isfinite(eta) or eta > 1e6:
            break

    monotone = all(seq[i+1] <= seq[i] for i in range(len(seq)-1))
    S = float(sum(seq))

    # Products with C_theta = theta / eta0 → factors start at (1 - theta) > 0.
    thetas = [0.1, 0.5, 0.9]
    prod_map: Dict[str, float] = {}
    for theta in thetas:
        C = theta / eta0 if eta0 > 0 else float('inf')
        logP = 0.0
        positive = True
        for x in seq:
            term = 1.0 - C * x
            if term <= 0.0:
                positive = False
                logP = float('-inf')
                break
            logP += math.log(term)
        P = math.exp(logP) if positive else 0.0
        prod_map[f"P_theta={theta}"] = P
        prod_map[f"P_theta={theta}_positive"] = positive

    # Double-exponential signature proxy:
    # slope over k of log(log(1/eta_k)) once eta_k < 0.1
    logs: List[float] = []
    ks: List[int] = []
    for i, x in enumerate(seq):
        if 0.0 < x < 0.1:
            try:
                logs.append(math.log(math.log(1.0 / x)))
                ks.append(i)
            except ValueError:
                pass
    slope = None
    if len(ks) >= 3:
        n = len(ks)
        mean_x = sum(ks) / n
        mean_y = sum(logs) / n
        num = sum((ks[i] - mean_x) * (logs[i] - mean_y) for i in range(n))
        den = sum((ks[i] - mean_x) ** 2 for i in range(n))
        slope = (num / den) if den > 0 else None

    finite_eff = [x for x in effA if math.isfinite(x)]
    result = {
        "name": cfg.name,
        "A": cfg.A,
        "alpha": cfg.alpha,
        "eta0": eta0,
        "steps_computed": len(seq) - 1,
        "monotone_decreasing": monotone,
        "sum_eta": S,
        "min_eta": float(min(seq)),
        "max_effA": float(max(finite_eff)) if finite_eff else float('nan'),
        "mean_effA": float(sum(finite_eff) / len(finite_eff)) if finite_eff else float('nan'),
        "vary_A": cfg.vary_A,
        "rho": cfg.rho,
        "eps": cfg.eps,
        "double_exp_slope": slope if slope is not None else float('nan'),
    }
    result.update(prod_map)
    return seq, Ak_list, effA, result

def main():
    # Output directory
    out_dir = Path(__file__).resolve().parent / "qc_outputs"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Scenario grid
    A_values = [0.5, 1, 5, 50, 500]
    alphas = [0.5, 0.9, 0.99]

    scenarios: List[ScenarioConfig] = []

    # Ideal (no variability, no leak)
    for A in A_values:
        for a in alphas:
            scenarios.append(ScenarioConfig(name=f"Ideal A={A}, alpha={a}", A=A, alpha=a))

    # Variable A_k (rho = 0.25)
    for A in [1, 5, 50]:
        for a in [0.9, 0.99]:
            scenarios.append(ScenarioConfig(name=f"VarA A={A}, alpha={a}, rho=0.25",
                                            A=A, alpha=a, vary_A=True, rho=0.25))

    # Linear leak tests
    for eps in [0.05, 0.2, 0.5, 0.9]:
        scenarios.append(ScenarioConfig(name=f"Leak eps={eps}, A=5, alpha=0.99",  A=5,  alpha=0.99, eps=eps))
        scenarios.append(ScenarioConfig(name=f"Leak eps={eps}, A=50, alpha=0.99", A=50, alpha=0.99, eps=eps))

    # Run everything
    records: List[Dict] = []
    trajectories: Dict[str, List[float]] = {}
    leak_trajectories: Dict[str, List[float]] = {}

    for cfg in scenarios:
        seq, Ak_list, effA, rec = run_sequence(cfg)
        records.append(rec)
        # Save a few representative ideal trajectories for plotting
        if cfg.eps == 0.0 and not cfg.vary_A and cfg.alpha == 0.99 and cfg.A in [1, 5, 50, 500]:
            trajectories[cfg.name] = seq
        # Save leak trajectories (A=50 for side-by-side)
        if cfg.eps > 0.0 and cfg.A in [5, 50] and cfg.alpha == 0.99:
            leak_trajectories[cfg.name] = seq

    # DataFrames
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by=["eps", "vary_A", "A", "alpha"]).reset_index(drop=True)

    leak_df = df[df["eps"] > 0.0][
        ["name","A","alpha","eps","eta0","steps_computed","monotone_decreasing","sum_eta",
         "P_theta=0.9_positive","P_theta=0.9"]
    ].copy().reset_index(drop=True)

    # Save CSVs
    df.to_csv(out_dir / "stress_test_summary.csv", index=False)
    leak_df.to_csv(out_dir / "leak_scenarios_summary.csv", index=False)

    # Plot 1: Ideal contraction trajectories (log scale)
    plt.figure()
    for name, seq in trajectories.items():
        plt.semilogy(range(len(seq)), seq, label=name)
    plt.xlabel("k (RG step)")
    plt.ylabel("eta_k")
    plt.title("Ideal quadratic contraction: eta_k vs k (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ideal_trajectories.png", dpi=180)
    plt.close()

    # Plot 2: Leak comparison (quadratic vs quadratic+linear, A=50 baseline)
    plt.figure()
    baseline_name = "Ideal A=50, alpha=0.99"
    if baseline_name in trajectories:
        plt.semilogy(range(len(trajectories[baseline_name])), trajectories[baseline_name], label=baseline_name)
    for name, seq in leak_trajectories.items():
        if "A=50" in name:
            plt.semilogy(range(len(seq)), seq, label=name)
    plt.xlabel("k (RG step)")
    plt.ylabel("eta_k")
    plt.title("Effect of linear leak: eta_{k+1} = A eta_k^2 + eps eta_k (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "leak_effect.png", dpi=180)
    plt.close()

    # Console preview
    print(f"\nSaved CSVs and figures to: {out_dir}\n")
    print("First 12 rows (all scenarios):")
    with pd.option_context('display.width', 140, 'display.max_columns', 20):
        print(df.head(12).to_string(index=False))
    print("\nLeak scenarios:")
    with pd.option_context('display.width', 140, 'display.max_columns', 20):
        print(leak_df.to_string(index=False))

if __name__ == "__main__":
    main()