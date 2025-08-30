# Quadratic Contraction Lemma — “No Linear Terms Survive”
*A referee-facing repo with proofs, stress tests, and cross-fit validation.*

This folder contains:

- A rigorous, self-contained derivation of why the **linear term vanishes** (the order-1 Ursell/cumulant is zero after local extraction/centering).
- A proof of the **Quadratic Contraction Lemma**: for the KP norm of post-RG activities,
  $$
  \eta_{k+1} \le A\,\eta_k^2 \quad\text{with }A\text{ geometry-only}.
  $$
- Reproducible **numerical evidence**:
  1. Pure quadratic dynamics vs. a **linear leak**.
  2. A **cumulant** test (centered vs. uncentered).
  3. A **cross-fit** test with z-scores and a Hessian≈Covariance check.

The results are packaged as figures/CSVs and compiled into a LaTeX report.

---

## Quick Start

### 1. Python Environment Setup

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib
```

### 2. Run the Quadratic Recursion Stress Tests

This script tests the stability of the `η_k` recurrence.

**How to Run:**
```bash
python3 contraction_stability_analyzer_v2.py run --file scenarios.json
```
*(Requires a `scenarios.json` file, see section 5 for an example)*

**Outputs (in `contraction_analysis_results/`):**
- `eta_k_decay.png`: Shows that `η_k` decays double-exponentially under a pure quadratic flow.
- `loss_product.png`: Shows that the cumulative loss product $\prod (1 - C\,\eta_j)$ remains strictly positive for ideal scenarios but vanishes if a linear leak is present.
- `results_summary.csv`: A table with detailed metrics for all scenarios.

**Interpretation:**
- **PASS:** Ideal runs show `is_summable: True`. This means the loss product is greater than zero, providing numerical support for the survival of the string tension.
- **FAIL (by design):** Scenarios with a linear leak (`eps > 0`) show `is_summable: False`, demonstrating the fragility of the proof to such a term.

### 3. Run the Cross-Fit “No Linear Term” Validation

This script provides referee-grade numerical evidence that the centering procedure eliminates linear terms.

**How to Run:**
```bash
python3 no_linear_terms_survive_crosssplit.py
```

**Outputs (in `cv_outputs/`):**
- `zscore_per_coordinate.png`: A plot showing the z-scores (`|gradient / std_error|`) for each coordinate. All should be below the significance threshold (e.g., 3.0).
- `summary.json`: A one-line verdict with the max z-score and the Hessian vs. Covariance error.

**Interpretation:**
- **PASS:** A `PASS` in `summary.json` (e.g., `z_max_direct <= 3.0`) means the residual gradient is statistically indistinguishable from zero. This provides strong numerical support that the linear term vanishes.

-----

## Theory in a Nutshell

- **Extraction/Centering:** An operator is split into its expectation and a centered remainder, $begin:math:text$K = \\mathbb{E}[K] + \\widetilde K$end:math:text$, where by construction $begin:math:text$\\mathbb{E}[\\widetilde K]=0$end:math:text$.
- **Connected Expansion (BKAR/Ursell):** The post-RG activity is a sum over connected clusters of centered fine-scale operators:
  $$
  K_{k+1}(\Gamma') \;=\; \sum_{n\ge 1}\frac{1}{n!}\!\!\sum_{\Gamma_1,\dots,\Gamma_n}
  \Phi_T(\Gamma_1,\dots,\Gamma_n)\,\prod_i \widetilde K(\Gamma_i).
  $$
  The $begin:math:text$n=1$end:math:text$ term is the first cumulant, which is $begin:math:text$\\mathbb{E}[\\widetilde K]=0$end:math:text$. Therefore, the expansion has **no linear term** and starts at the quadratic ($begin:math:text$n=2$end:math:text$) level.
- **Quadratic Bound:** The tree-graph inequality for the $begin:math:text$n\\ge 2$end:math:text$ terms, combined with combinatorial counting and the finite-range geometry of the RG, yields the quadratic contraction $begin:math:text$\\eta_{k+1}\\le A\\,\\eta_k^2$end:math:text$, where $begin:math:text$A$end:math:text$ depends only on the geometry.

-----

## Configuration File (`scenarios.json`)

To run batch experiments, create a `scenarios.json` file. It should contain a list of scenario objects.

**Example `scenarios.json`:**
```json
[
  {
    "name": "Ideal (A=50, Strong Seed)",
    "A": 50.0,
    "eta0": 0.01
  },
  {
    "name": "Noisy A (rho=0.25)",
    "A": 50.0,
    "eta0": 0.01,
    "rho": 0.25
  },
  {
    "name": "Fatal Leak (eps=0.05)",
    "A": 50.0,
    "eta0": 0.01,
    "eps": 0.05
  }
]
```

**Parameters:**
- `name` (str): A descriptive name for the scenario.
- `A` (float): The ideal quadratic contraction constant.
- `eta0` (float): The initial value of the norm at the seed scale, `η₀`.
- `steps` (int, optional): The number of RG steps to simulate.
- `eps` (float, optional): The “linear leak” coefficient. A small positive value (`> 0`) simulates a failure of pointwise centering.
- `rho` (float, optional): The fractional noise on `A`, simulating variability in the RG map.
- `C_loss_factor` (float, optional): The geometric constant `C` in the multiplicative loss factor `(1 - C η_k)`.

-----

## One-line Summary for Referees

The RG step uses centered cumulants, so the order-1 Ursell coefficient is exactly zero (no linear term). BKAR tree bounds, combined with KP smallness and finite-range geometry, imply the quadratic contraction $begin:math:text$\\eta_{k+1}\\le A\\,\\eta_k^2$end:math:text$ with a scale-independent constant $begin:math:text$A$end:math:text$. The included numerics illustrate both facts: centered gradients are statistically zero (cross-fit z-scores ≤ 3), and any injected linear component destroys the contraction required for summability.
