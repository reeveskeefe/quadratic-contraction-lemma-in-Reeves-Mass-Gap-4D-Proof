#!/usr/bin/env python3
"""
“No linear terms survive” checker via cumulant-generating function diagnostics.

Key fact:
  For centered activities X~μ, the cumulant generating function
      L(J) := log E_mu[ exp(J · X) ]
  satisfies:
      ∇L(0) = E[X] = 0                (no linear term),
      ∇^2 L(0) = Cov(X)               (quadratic term is the first non-zero).
Thus, expansions of connected (Ursell) parts begin at order n >= 2.

This script:
  1) Constructs correlated, non-Gaussian activities (to avoid any Gaussian-only artifacts),
  2) Centers them to obtain X_tilde = X - E[X],
  3) Uses symmetric finite differences around J=0 to estimate:
       - gradient ∇L(0)  (should be ~0 after centering)
       - Hessian ∇^2L(0) (should match sample covariance)
  4) Repeats the same test WITHOUT centering to show a “linear leak” appears.

Outputs (in ./no_linear_outputs):
  - gradient_centered.csv
  - gradient_uncentered.csv
  - hessian_centered_estimate.csv
  - covariance_centered.csv
  - probe_directions.png   (gradient along random directions: centered vs uncentered)
  - summary.txt            (readable PASS/FAIL summary)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Config (feel free to tweak)
# ----------------------------
RNG_SEED         = 42
DIM              = 20          # number of activities in the block (dim of J)
N_SAMPLES        = 200_000     # Monte Carlo samples
FD_H             = 5e-3        # finite-difference step for L(J)
N_DIRS           = 16          # random directions to test gradient-on-directions
TOL_GRAD_LINF    = 5e-3        # ||∇L(0)||_inf tolerance for “PASS”
TOL_GRAD_DIR     = 3e-3        # |d/dε L(εv)| at ε=0 along random directions (PASS if below)
TOL_HESS_FRO     = 2e-2        # relative Frobenius error between Hessian and Cov (PASS if below)

# ----------------------------
# Helpers
# ----------------------------
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def build_non_gaussian_correlated(dim, n, rng):
    """
    Build non-Gaussian, correlated data:
      Z ~ N(0, I)
      Y = A Z  (A: random linear mix to induce correlation)
      X = tanh(Y) + 0.2 * (Y**3) + 0.1 * softplus(2Y) * sign(Y)
    This produces skew/kurtosis and strong correlations.
    """
    # Random mixing matrix with decent condition
    M = rng.normal(size=(dim, dim))
    A, _ = np.linalg.qr(M)      # orthonormal base
    s = np.linspace(1.0, 3.0, dim)
    A = A @ np.diag(s)          # scale singular values
    Z = rng.normal(size=(dim, n))
    Y = A @ Z
    X = np.tanh(Y) + 0.2*(Y**3) + 0.1*softplus(2*Y)*np.sign(Y)
    return X.T  # shape (n, dim)

def log_mgf(X, J):
    """
    Estimate L(J) = log E[exp(J·X)] from samples X (n x d).
    Uses log-sum-exp stabilization.
    """
    dots = X @ J  # shape (n,)
    m = np.max(dots)
    return m + np.log(np.mean(np.exp(dots - m)))

def grad_at_zero_fd(X_centered, h):
    """
    ∇L(0) via symmetric finite differences coordinatewise:
      ∂_i L(0) ≈ [ L(h e_i) - L(-h e_i) ] / (2h)
    """
    n, d = X_centered.shape
    grad = np.zeros(d)
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0
        Lp = log_mgf(X_centered,  h*e)
        Lm = log_mgf(X_centered, -h*e)
        grad[i] = (Lp - Lm) / (2*h)
    return grad

def hessian_at_zero_fd(X_centered, h):
    """
    ∇^2 L(0) via symmetric 2D finite differences:
      H_ij ≈ [ L(h e_i + h e_j) - L(h e_i - h e_j)
               - L(-h e_i + h e_j) + L(-h e_i - h e_j) ] / (4 h^2)
    Uses symmetry to fill the matrix.
    """
    d = X_centered.shape[1]
    H = np.zeros((d, d))
    basis = np.eye(d)
    # Diagonals first (i=j) using 1D second derivative trick for better stability
    for i in range(d):
        e = basis[i]
        Lp = log_mgf(X_centered,  h*e)
        L0 = log_mgf(X_centered,  0*e)
        Lm = log_mgf(X_centered, -h*e)
        H[i, i] = (Lp - 2*L0 + Lm) / (h*h)
    # Off-diagonals
    for i in range(d):
        ei = basis[i]
        for j in range(i+1, d):
            ej = basis[j]
            Lpp = log_mgf(X_centered,  h*(ei + ej))
            Lpm = log_mgf(X_centered,  h*(ei - ej))
            Lmp = log_mgf(X_centered,  h*(-ei + ej))
            Lmm = log_mgf(X_centered,  h*(-ei - ej))
            H_ij = (Lpp - Lpm - Lmp + Lmm) / (4*h*h)
            H[i, j] = H[j, i] = H_ij
    return H

def grad_along_directions(X, h, dirs):
    """
    For unit directions v_k, estimate d/dε L(ε v_k)|_{0} via symmetric FD:
      ≈ [L(h v) - L(-h v)]/(2h)
    """
    vals = []
    for v in dirs:
        v = v / np.linalg.norm(v)
        Lp = log_mgf(X,  h*v)
        Lm = log_mgf(X, -h*v)
        vals.append((Lp - Lm) / (2*h))
    return np.array(vals)

# ----------------------------
# Main
# ----------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    out = Path(__file__).resolve().parent / "no_linear_outputs"
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build raw activities (non-Gaussian, correlated)
    X_raw = build_non_gaussian_correlated(DIM, N_SAMPLES, rng)  # shape (n, d)

    # 2) Center them (this mirrors the “cumulant / centered activity” step)
    mean_raw = X_raw.mean(axis=0, keepdims=True)
    X_ctr = X_raw - mean_raw  # centered
    cov_ctr = np.cov(X_ctr, rowvar=False)

    # 3) Gradient/Hessian at zero for centered data
    grad_ctr = grad_at_zero_fd(X_ctr, FD_H)
    H_ctr = hessian_at_zero_fd(X_ctr, FD_H)

    # 4) Negative control: without centering (should show linear leak)
    grad_unc = grad_at_zero_fd(X_raw, FD_H)

    # 5) Random-direction checks (strongest “no linear term” signal)
    dirs = rng.normal(size=(N_DIRS, DIM))
    dir_grads_ctr = grad_along_directions(X_ctr, FD_H, dirs)   # should be ~0
    dir_grads_unc = grad_along_directions(X_raw, FD_H, dirs)   # should be noticeably ≠0

    # 6) Metrics & PASS/FAIL
    grad_ctr_linf = np.max(np.abs(grad_ctr))
    rel_fro = np.linalg.norm(H_ctr - cov_ctr, 'fro') / (np.linalg.norm(cov_ctr, 'fro') + 1e-12)
    dir_ctr_max = np.max(np.abs(dir_grads_ctr))

    pass_grad = grad_ctr_linf <= TOL_GRAD_LINF
    pass_dir  = dir_ctr_max     <= TOL_GRAD_DIR
    pass_hess = rel_fro         <= TOL_HESS_FRO

    # 7) Save tables
    pd.DataFrame({"grad_centered": grad_ctr}).to_csv(out / "gradient_centered.csv", index=False)
    pd.DataFrame({"grad_uncentered": grad_unc}).to_csv(out / "gradient_uncentered.csv", index=False)
    pd.DataFrame(H_ctr).to_csv(out / "hessian_centered_estimate.csv", index=False, header=False)
    pd.DataFrame(cov_ctr).to_csv(out / "covariance_centered.csv", index=False, header=False)

    # 8) Plot direction-derivative comparison
    plt.figure()
    idx = np.arange(N_DIRS)
    width = 0.4
    plt.bar(idx - width/2, dir_grads_ctr, width, label="Centered (expected ~0)")
    plt.bar(idx + width/2, dir_grads_unc, width, label="Uncentered (linear leak)")
    plt.axhline(0.0)
    plt.xlabel("Random direction index")
    plt.ylabel("Directional derivative at 0")
    plt.title("Directional gradient of L at J=0: centered vs uncentered")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "probe_directions.png", dpi=180)
    plt.close()

    # 9) Human-readable summary
    summary = []
    summary.append("=== No-Linear-Terms Survive: Diagnostic Summary ===")
    summary.append(f"Samples: {N_SAMPLES:,} | Dim: {DIM} | FD step h={FD_H}")
    summary.append("")
    summary.append("Centered case (should have NO linear term):")
    summary.append(f"  ||∇L(0)||_∞ = {grad_ctr_linf:.3e}  (tolerance {TOL_GRAD_LINF:.1e})  -> {'PASS' if pass_grad else 'FAIL'}")
    summary.append(f"  Max |directional grad| over {N_DIRS} dirs = {dir_ctr_max:.3e}  (tol {TOL_GRAD_DIR:.1e}) -> {'PASS' if pass_dir else 'FAIL'}")
    summary.append(f"  Hessian vs Covariance (Frobenius rel. err) = {rel_fro:.3e}  (tol {TOL_HESS_FRO:.1e}) -> {'PASS' if pass_hess else 'FAIL'}")
    summary.append("")
    summary.append("Uncentered control (SHOULD show a linear leak):")
    summary.append(f"  ||∇L(0)||_∞ (uncentered) = {np.max(np.abs(grad_unc)):.3e}  -> should be noticeably > 0")
    summary.append("")
    overall = "PASS" if (pass_grad and pass_dir and pass_hess) else "FAIL"
    summary.append(f"OVERALL: {overall}")
    text = "\n".join(summary)
    print(text)
    (out / "summary.txt").write_text(text)

if __name__ == "__main__":
    main()