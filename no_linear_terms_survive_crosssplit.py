#!/usr/bin/env python3
"""
Cross-fit proof-of-principle: "no linear term survives" numerically,
with scale-aware SE/z-score checks + optional antithetic demo.

PASS criteria:
  - max |z_j| = |g_j| / SE_j ≤ 3   for direct mean gradient on B (cross-fit)
  - Hessian vs Cov rel. Frobenius ≤ 2e-2
FD gradient is also computed and should match the direct mean.

Optionally, ANTITHETIC=True runs an odd-only generator with y↦f(y),−y↦f(−y),
forcing ∇L(0)=0 up to machine precision (illustration of exact centering).
"""

import math, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
RNG_SEED   = 42
DIM        = 20
N_SAMPLES  = 200_000       # total samples
K_FOLDS    = 10            # cross-fit folds (use complements as "A" for each test fold)
FD_H       = 5e-3
TRIM_FRAC  = 0.02          # trim on training(A) for robust mean
WINSOR_FRAC= 0.02          # winsorize test(B) to A-quantiles
TOL_Z_MAX  = 3.0
TOL_HFRO   = 2e-2
ANTITHETIC = False         # set True for the "exact-zero" demo

# ------------- Data generators ----------
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)

def build_heavy_tailed(dim, n, rng):
    """Non-Gaussian correlated with heavy-ish tails."""
    M = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(M)
    s = np.linspace(1.0, 3.0, dim)
    A = Q @ np.diag(s)
    Z = rng.normal(size=(dim, n))
    Y = A @ Z
    # mix odd + non-odd parts to make it realistic/hard
    X = np.tanh(Y) + 0.2*(Y**3) + 0.05*softplus(2*Y)
    return X.T  # (n,d)

def build_antithetic_odd(dim, n_pairs, rng):
    """Odd-only f + antithetic pairs -> exact E[X]=0."""
    M = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(M)
    s = np.linspace(1.0, 3.0, dim)
    A = Q @ np.diag(s)
    Z = rng.normal(size=(dim, n_pairs))
    Y = A @ Z
    fY = np.tanh(Y) + 0.2*(Y**3)     # strictly odd
    X1 = fY.T                        # (n_pairs, d)
    X2 = (-fY).T                     # antithetic
    X  = np.vstack([X1, X2])
    rng.shuffle(X, axis=0)
    return X

# ------------- Robust A→B centering ----
def trimmed_mean(A, frac):
    if frac <= 0.0: return A.mean(axis=0)
    lo = np.quantile(A, frac, axis=0)
    hi = np.quantile(A, 1.0-frac, axis=0)
    mu = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        aj = A[:, j]
        m  = (aj[(aj>=lo[j])&(aj<=hi[j])]).mean()
        mu[j] = m
    return mu

def winsorize_to_A(A, B, frac):
    if frac <= 0.0: return B
    lo = np.quantile(A, frac, axis=0)
    hi = np.quantile(A, 1.0-frac, axis=0)
    return np.clip(B, lo, hi)

# ------------- CGF utils ---------------
def log_mgf(X, J):
    dots = X @ J; m = float(np.max(dots))
    return m + math.log(np.mean(np.exp(dots - m)))

def fd_grad_at_zero(X, h):
    d = X.shape[1]; g = np.zeros(d); I = np.eye(d)
    for i in range(d):
        e = I[i]; Lp = log_mgf(X, h*e); Lm = log_mgf(X, -h*e)
        g[i] = (Lp - Lm) / (2*h)
    return g

def fd_hessian_at_zero(X, h):
    d = X.shape[1]; H = np.zeros((d,d)); I = np.eye(d)
    L0 = log_mgf(X, np.zeros(d))
    for i in range(d):
        e = I[i]; Lp = log_mgf(X, h*e); Lm = log_mgf(X, -h*e)
        H[i,i] = (Lp - 2*L0 + Lm) / (h*h)
    for i in range(d):
        ei = I[i]
        for j in range(i+1, d):
            ej = I[j]
            Lpp=log_mgf(X, h*(ei+ej)); Lpm=log_mgf(X, h*(ei-ej))
            Lmp=log_mgf(X, h*(-ei+ej));Lmm=log_mgf(X, h*(-ei-ej))
            H[i,j]=(Lpp - Lpm - Lmp + Lmm)/(4*h*h)
            H[j,i]=H[i,j]
    return H

# ------------- Cross-fitting -----------
def cross_fit_gradients(X, k_folds, trim_frac, winsor_frac, h, rng):
    n, d = X.shape
    idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)
    fold_grad_direct = []
    fold_grad_fd     = []
    for t_idx in folds:
        a_idx = np.setdiff1d(idx, t_idx, assume_unique=False)
        A = X[a_idx]; B = X[t_idx]
        muA = trimmed_mean(A, trim_frac)
        B_w = winsorize_to_A(A, B, winsor_frac)
        Bc  = B_w - muA
        # direct == sample mean (exact gradient at 0)
        g_direct = Bc.mean(axis=0)
        g_fd     = fd_grad_at_zero(Bc, h)
        fold_grad_direct.append(g_direct)
        fold_grad_fd.append(g_fd)
    Gd = np.vstack(fold_grad_direct)   # (k,d)
    Gf = np.vstack(fold_grad_fd)       # (k,d)
    # aggregate
    g_direct = Gd.mean(axis=0)
    g_fd     = Gf.mean(axis=0)
    # SE via fold variability (conservative, accounts for both halves)
    se_direct = Gd.std(axis=0, ddof=1)/math.sqrt(k_folds)
    se_fd     = Gf.std(axis=0, ddof=1)/math.sqrt(k_folds)
    return g_direct, se_direct, g_fd, se_fd, Gd, Gf

# ------------- Main --------------------
def main():
    out = Path(__file__).resolve().parent / "cv_outputs"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    if ANTITHETIC:
        X = build_antithetic_odd(DIM, N_SAMPLES//2, rng)
    else:
        X = build_heavy_tailed(DIM, N_SAMPLES, rng)

    g_dir, se_dir, g_fd, se_fd, Gd, Gf = cross_fit_gradients(
        X, K_FOLDS, TRIM_FRAC, WINSOR_FRAC, FD_H, rng
    )

    # z-scores and decisions
    eps = 1e-12
    z_dir = np.abs(g_dir) / np.maximum(se_dir, eps)
    z_fd  = np.abs(g_fd)  / np.maximum(se_fd,  eps)
    zmax  = float(np.max(z_dir))
    pass_z = (zmax <= TOL_Z_MAX)

    # Hessian vs Cov on the *pooled, centered* data using the fold means
    # (approximate check on one random fold’s centered data to keep it cheap)
    # Take the first fold again:
    n, d = X.shape
    idx = np.arange(n); rng.shuffle(idx)
    t_idx = np.array_split(idx, K_FOLDS)[0]
    a_idx = np.setdiff1d(idx, t_idx, assume_unique=False)
    A = X[a_idx]; B = X[t_idx]
    muA = trimmed_mean(A, TRIM_FRAC)
    B_w = winsorize_to_A(A, B, WINSOR_FRAC)
    Bc  = B_w - muA
    H   = fd_hessian_at_zero(Bc, FD_H)
    Cov = np.cov(Bc, rowvar=False)
    rel_fro = float(np.linalg.norm(H - Cov, 'fro') / (np.linalg.norm(Cov, 'fro') + eps))
    pass_h = (rel_fro <= TOL_HFRO)

    overall = "PASS" if (pass_z and pass_h) else "FAIL"

    # Save artifacts
    pd.DataFrame({"g_direct": g_dir, "se_direct": se_dir, "z_direct": z_dir}).to_csv(out/"crossfit_gradient_direct.csv", index=False)
    pd.DataFrame({"g_fd": g_fd, "se_fd": se_fd, "z_fd": z_fd}).to_csv(out/"crossfit_gradient_fd.csv", index=False)
    pd.DataFrame(Gd).to_csv(out/"fold_gradients_direct.csv", index=False, header=False)
    pd.DataFrame(Gf).to_csv(out/"fold_gradients_fd.csv", index=False, header=False)
    pd.DataFrame(H).to_csv(out/"hessian_example.csv", index=False, header=False)
    pd.DataFrame(Cov).to_csv(out/"covariance_example.csv", index=False, header=False)

    # Tiny plot: coordinate-wise z-scores (direct)
    plt.figure()
    # Older/newer Matplotlib-safe stem plot (no 'use_line_collection' kwarg)
    markerline, stemlines, baseline = plt.stem(range(len(z_dir)), z_dir)
    plt.axhline(TOL_Z_MAX, linestyle="--")
    plt.title("Cross-fit z-scores per coordinate (direct gradient)")
    plt.xlabel("coordinate")
    plt.ylabel("|z|")
    plt.tight_layout()
    plt.savefig(out/"zscore_per_coordinate.png", dpi=180)
    plt.close()

    # Summary
    summary = {
        "RNG_SEED": RNG_SEED, "DIM": DIM, "N_SAMPLES": N_SAMPLES, "K_FOLDS": K_FOLDS,
        "FD_H": FD_H, "TRIM_FRAC": TRIM_FRAC, "WINSOR_FRAC": WINSOR_FRAC, "ANTITHETIC": ANTITHETIC,
        "z_max_direct": zmax, "z_tol": TOL_Z_MAX, "rel_fro(H,Cov)": rel_fro,
        "rel_fro_tol": TOL_HFRO, "OVERALL": overall
    }
    print(json.dumps(summary, indent=2))
    (out/"summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()