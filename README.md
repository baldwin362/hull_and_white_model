# Hull & White Stochastic Volatility Model

Numerical reproduction of the experiments from **Hull & White (1987)** — *"The Pricing of Options on Assets with Stochastic Volatilities"*, Journal of Finance.

The project implements two pricing methods for European call options under stochastic volatility, and reproduces all tables and figures from the paper.

---

## Model

The stock price and its variance follow correlated geometric Brownian motions:

```
dS =  μ S dt  +  √V · S dW₁
dV =  ξ V dW₂

d⟨W₁, W₂⟩ = ρ dt
```

| Parameter | Description |
|-----------|-------------|
| `σ₀`      | Initial volatility (`V₀ = σ₀²`) |
| `ξ`       | Volatility of volatility |
| `ρ`       | Correlation between stock and variance Brownian motions |
| `μ`       | Drift of variance (0 in most experiments) |
| `T`       | Option maturity |
| `r`       | Risk-free rate |

### Pricing methods

**Series solution (Eq. 9)** — analytical two-term expansion around the Black-Scholes price, correcting for the variance and skewness of `V̄`. Only valid when `ρ = 0`.

**Monte Carlo** — two simulation schemes depending on `ρ`:
- `ρ = 0`: simulate only the variance path, price with `E[C_BS(√V̄)]` (fast, no discretisation of `S`)
- `ρ ≠ 0`: simulate `(S, V)` jointly step-by-step via Cholesky decomposition

Both schemes use **antithetic variates** on the stock Brownian motion for variance reduction.

---

## Project Structure

```
hull_and_white_model/
│
├── src/                                      # Production scripts
│   ├── comparaison_serie_monte_carlo.py      # Table I  — MC vs series solution
│   ├── biais_en_fonction_de_xhi.py           # Figure 1 — B-S vs true price (bias ×25)
│   ├── biais_en_fonction_volatilite.py       # Figure 2 — bias vs σ₀
│   ├── biais_en_fonction_de_xi.py            # Figure 3 — bias vs ξ
│   ├── biais_en_fonction_rho.py              # Figure    — bias vs ρ (Table II as graph)
│   ├── price_bias_vs_maturity.py             # Figure 4  — bias vs maturity T
│   ├── table_II.py                           # Table II  — bias (%, ρ × S/X × T)
│   ├── tableIII.py                           # Table III — implied volatilities
│   ├── tableIII_omar.py                      # Table III — improved (mc_rho0 for ρ=0, N↑)
│   ├── smile_de_volatilite.py                # Volatility smiles (ρ, ξ, T effects)
│   ├── smile_de_volatilite_omar.py           # Improved smile for ρ effect
│   └── convergence_analysis_omar.py          # MC convergence analysis
│
├── src/<script>/                             # Each script auto-saves output here
│   ├── table_I.txt
│   ├── table_II.txt
│   ├── table_III.txt
│   ├── figure_1.png  …  figure_4.png
│   ├── smile_*.png
│   └── convergence_*.png
│
├── test_experiments/                         # Exploratory scripts (exp1 … exp10)
│
└── experiments_in_the_article/               # Screenshots of the original paper
```

---

## Experiments & Results

### Table I — MC vs Series Solution
`src/comparaison_serie_monte_carlo.py`

Parameters: `σ₀ = 10%`, `ξ = 1`, `ρ = 0`, `T = 180d`, `r = 0`.
Compares Black-Scholes, Eq. 9, and MC prices across `S/X ∈ [0.75, 1.25]`.
→ Validates that the series solution captures the B-S bias accurately.

---

### Figure 1 — B-S vs True Price
`src/biais_en_fonction_de_xhi.py`

Parameters: `σ₀ = 15%`, `ξ = 1`, `ρ = 0`, `T = 180d`.
Overlays B-S and Eq. 9 prices with bias exaggerated ×25. Shows the sign-change structure of the bias.

---

### Figure 2 — Effect of `σ₀`
`src/biais_en_fonction_volatilite.py`

Varies `σ₀ ∈ {10%, 15%, 20%}` with `ξ = 1`, `ρ = 0`.
Higher initial volatility *shrinks* the relative bias.

---

### Figure 3 — Effect of `ξ`
`src/biais_en_fonction_de_xi.py`

Varies `ξ ∈ {1, 2, 3}` with `σ₀ = 15%`, `ρ = 0`.
Higher `ξ` amplifies the bias dramatically (Jensen effect on the convex B-S formula).

---

### Figure 4 — Effect of Maturity
`src/price_bias_vs_maturity.py`

Varies `T ∈ {45, 90, 135d}` with `r = 10%`, `σ₀ = 15%`, `ξ = 1`, `ρ = 0`.
Longer maturities give stochastic volatility more time to deviate, increasing the bias.

---

### Table II — Price Bias vs `ρ`, `S/X`, `T`
`src/table_II.py`

Full sensitivity matrix: `S/X ∈ {0.90…1.10}`, `ρ ∈ {−1, −0.5, 0, 0.5, 1}`, `T ∈ {90, 180, 270d}`.
Reports bias (%) and standard error for each combination.

---

### Table III — Implied Volatility
`src/tableIII_omar.py`

Back-solves implied volatility from the MC prices in Table II via bisection.
Uses the efficient `mc_rho0` scheme for `ρ = 0` (N = 100 000) and the joint scheme for `ρ ≠ 0` (N = 50 000).

Expected values at `ρ = 0`, `T = 180d`:

| S/X | σ_imp (%) |
|-----|-----------|
| 0.90 | ≈ 15.03 |
| 0.95 | ≈ 14.81 |
| 1.00 | ≈ 14.72 |
| 1.05 | ≈ 14.77 |
| 1.10 | ≈ 14.94 |

---

### Volatility Smiles
`src/smile_de_volatilite_omar.py` (ρ effect) · `src/smile_de_volatilite.py` (ξ, T effects)

| Figure | What varies | Key result |
|--------|-------------|------------|
| `smile_reference.png` | — | Flat smile at 15% when `ρ = 0`, `ξ = 1` |
| `smile_effet_xi.png` | `ξ ∈ {0.5, 1, 2}` | Larger `ξ` → deeper symmetric U-shape |
| `smile_effet_rho.png` | `ρ ∈ {−1,−0.5, 0, 0.5, 1}` | Negative `ρ` → classic equity skew/smirk |
| `smile_effet_T.png` | `T ∈ {90, 180, 270d}` | Longer `T` → flatter smile |

---

### MC Convergence Analysis
`src/convergence_analysis_omar.py`

Three outputs (seed = 42, ATM option, `σ₀ = 10%`, `ξ = 1`, `ρ = 0`):

| Output | Content |
|--------|---------|
| `convergence_nsims.png` | SE vs N (log-log), standard vs antithetic + `1/√N` reference |
| `convergence_nsteps.png` | MC price vs n (log-x), with Eq. 9 and B-S reference lines |
| `variance_reduction.txt` | Antithetic variance reduction factor ≈ **33×** at N = 50 000 |

---

## Getting Started

**Requirements:** Python ≥ 3.12, NumPy, SciPy, Matplotlib.

```bash
pip install numpy scipy matplotlib
```

Run any script directly — each one creates its own output subdirectory under `src/`:

```bash
python src/tableIII_omar.py            # Table III (implied vols)
python src/smile_de_volatilite_omar.py # Smile — effect of rho
python src/convergence_analysis_omar.py # Convergence analysis
```

---

## Reference

> Hull, J. C., & White, A. (1987). *The Pricing of Options on Assets with Stochastic Volatilities.* The Journal of Finance, 42(2), 281–300.
