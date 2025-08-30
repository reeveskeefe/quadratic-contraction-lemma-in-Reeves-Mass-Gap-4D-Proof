#!/usr/bin/env python3
"""
A research tool to analyze the stability of quadratic RG contractions.

This script simulates the recurrence eta_{k+1} = A_k * eta_k^2 + eps * eta_k
via a command-line interface.

Commands:
  - single: Run a single scenario with parameters from the command line.
  - run:    Run a batch of scenarios defined in a JSON configuration file.
"""
import math
import random
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Simulation and Analysis Engine (Logic is unchanged) ---

def run_simulation(config: dict) -> dict:
    """Runs a single scenario and returns a dictionary of results."""
    # Get parameters with defaults
    A = config["A"]
    eta0 = config["eta0"]
    steps = config.get("steps", 50)
    eps = config.get("eps", 0.0)
    rho = config.get("rho", 0.0)
    C_loss = config.get("C_loss_factor", 10.0)

    # Initialize
    eta = eta0
    eta_sequence = [eta0]
    loss_product_sequence = [1.0 - C_loss * eta0 if (1.0 - C_loss * eta0) > 0 else 0]

    for _ in range(steps):
        if eta < 1e-300: break

        A_k = A * (1.0 + random.uniform(-rho, rho)) if rho > 0 else A
        eta_next = A_k * (eta ** 2) + eps * eta
        eta_sequence.append(eta_next)
        
        loss_factor = 1.0 - C_loss * eta_next
        if loss_factor > 0 and loss_product_sequence[-1] > 0:
            loss_product_sequence.append(loss_product_sequence[-1] * loss_factor)
        else:
            loss_product_sequence.append(0)
            
        eta = eta_next

    # --- Analysis ---
    total_sum = sum(eta_sequence)
    final_prod = loss_product_sequence[-1]
    is_summable = final_prod > 1e-9

    log_logs = [math.log(math.log(1/e)) for e in eta_sequence if 0 < e < 0.1]
    decay_type = "None"
    if len(log_logs) > 5:
        if log_logs[-1] > log_logs[0] + 0.5 * (len(log_logs) - 1):
            decay_type = "Double-Exponential"
        else:
            decay_type = "Exponential"
    elif sum(1 for e in eta_sequence if e > 1e-9) < steps / 2:
        decay_type = "Exponential"

    return {
        "name": config["name"],
        "eta_sequence": eta_sequence,
        "loss_product_sequence": loss_product_sequence,
        "sum_eta": total_sum,
        "prod_losses": final_prod,
        "is_summable": is_summable,
        "decay_type": decay_type,
    }

def process_and_save_results(scenarios, out_dir_path):
    """Runs simulations for a list of scenarios and saves all outputs."""
    print(f"Running {len(scenarios)} scenario(s)...")
    results = [run_simulation(cfg) for cfg in scenarios]
    
    out_dir_path.mkdir(exist_ok=True, parents=True)
    
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if "sequence" not in k} for r in results])
    csv_path = out_dir_path / "results_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✔ Saved summary statistics to {csv_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for res in results:
        linestyle = "--" if res["name"].startswith("Fatal") else "-"
        plt.semilogy(res["eta_sequence"], label=res["name"], linestyle=linestyle, alpha=0.8)
    plt.title("RG Flow of the KP Norm (η_k) vs. RG Step (k)")
    plt.xlabel("k (RG Step)")
    plt.ylabel("η_k (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.ylim(bottom=1e-18)
    plt.tight_layout()
    plot1_path = out_dir_path / "eta_k_decay.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for res in results:
        linestyle = "--" if res["name"].startswith("Fatal") else "-"
        plt.plot(res["loss_product_sequence"], label=res["name"], linestyle=linestyle, alpha=0.8)
    plt.axhline(0, color='red', lw=1.5, linestyle=':')
    plt.title("Survival of String Tension vs. RG Step (k)")
    plt.xlabel("k (RG Step)")
    plt.ylabel("Cumulative Loss Product: Π(1 - Cη_j)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.ylim(bottom=-0.05)
    plt.tight_layout()
    plot2_path = out_dir_path / "loss_product.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()

    print(f"✔ Saved plots to {out_dir_path}")
    print("\nAnalysis complete.")

# --- Command-Line Interface ---

def main():
    parser = argparse.ArgumentParser(
        description="A research tool to analyze the stability of quadratic RG contractions."
    )
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- 'single' command ---
    parser_single = subparsers.add_parser("single", help="Run a single scenario from the command line.")
    parser_single.add_argument("--A", type=float, required=True, help="Quadratic contraction constant.")
    parser_single.add_argument("--eta0", type=float, required=True, help="Initial value of the norm.")
    parser_single.add_argument("--eps", type=float, default=0.0, help="Linear leak coefficient.")
    parser_single.add_argument("--rho", type=float, default=0.0, help="Fractional noise on A.")
    parser_single.add_argument("--steps", type=int, default=50, help="Number of RG steps.")
    parser_single.add_argument("--C_loss_factor", type=float, default=10.0, help="Constant in the loss factor (1 - C*eta).")
    parser_single.add_argument("--name", type=str, default="Single Run", help="Descriptive name for the run.")
    parser_single.add_argument("--outdir", type=str, default="contraction_analysis_results", help="Output directory.")

    # --- 'run' command ---
    parser_run = subparsers.add_parser("run", help="Run a batch of scenarios from a JSON file.")
    parser_run.add_argument("--file", type=str, required=True, help="Path to the JSON file with scenario definitions.")
    parser_run.add_argument("--outdir", type=str, default="contraction_analysis_results", help="Output directory.")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    scenarios_to_run = []
    output_directory = Path(args.outdir)

    if args.command == "single":
        scenarios_to_run.append(vars(args)) # vars() converts argparse.Namespace to dict
    elif args.command == "run":
        try:
            with open(args.file, 'r') as f:
                scenarios_to_run = json.load(f)
            if not isinstance(scenarios_to_run, list):
                raise ValueError("JSON file must contain a list of scenario objects.")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error reading scenarios file: {e}")
            return
            
    process_and_save_results(scenarios_to_run, output_directory)

if __name__ == "__main__":
    main()