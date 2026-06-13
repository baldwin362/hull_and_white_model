"""
Expérience 3 — Reproduction de la Table I du papier
MC vs solution en série (Eq. 9) vs B-S.
Paramètres : σ₀ = 10%, ξ = 1, μ = 0, T-t = 180 jours, r = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def series_solution_eq9(S, K, r, T, sigma, xi):
    """Solution en série (Eq. 9 du papier), cas μ=0, ρ=0.
    Retient les termes d'ordre 2 et 3 (variance et skewness de V_bar)."""
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    
    k = xi**2 * T
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nprime_d1 = norm.pdf(d1)
    
    # Terme 0 : prix Black-Scholes
    C0 = bs_call(S, K, r, T, sigma)
    
    if k < 1e-10:
        return C0
    
    # Terme 1 (correction variance, ordre 2)
    # Var(V_bar) = sigma^4 * [2(e^k - k - 1)/k^2 - 1]
    var_factor = 2.0 * (np.exp(k) - k - 1.0) / k**2 - 1.0
    C_second_deriv = S * np.sqrt(T) * nprime_d1 * (d1 * d2 - 1) / (4 * sigma**3)
    term1 = 0.5 * C_second_deriv * sigma**4 * var_factor
    
    # Terme 2 (correction skewness, ordre 3)
    # Skew factor from E[V_bar^3]
    skew_raw = (np.exp(3*k) - (9 + 18*k) * np.exp(k) + (8 + 24*k + 18*k**2 + 6*k**3)) / (3 * k**3)
    C_third_deriv = S * np.sqrt(T) * nprime_d1 * ((d1*d2 - 3)*(d1*d2 - 1) - (d1**2 + d2**2)) / (8 * sigma**5)
    term2 = (1.0/6.0) * C_third_deriv * sigma**6 * skew_raw
    
    return C0 + term1 + term2

def mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    """Monte Carlo Hull-White, cas ρ=0, avec variables antithétiques."""
    dt = T / n_steps
    prices = np.zeros(n_sims)
    
    for sim in range(n_sims):
        z = np.random.randn(n_steps)
        
        # Trajectoire directe
        V = np.zeros(n_steps + 1)
        V[0] = V0
        for i in range(n_steps):
            V[i+1] = V[i] * np.exp((mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * z[i])
        V_bar1 = np.mean(V)
        P1 = bs_call(S0, K, r, T, np.sqrt(max(V_bar1, 1e-12)))
        
        # Trajectoire antithétique
        V_anti = np.zeros(n_steps + 1)
        V_anti[0] = V0
        for i in range(n_steps):
            V_anti[i+1] = V_anti[i] * np.exp((mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * (-z[i]))
        V_bar2 = np.mean(V_anti)
        P2 = bs_call(S0, K, r, T, np.sqrt(max(V_bar2, 1e-12)))
        
        prices[sim] = (P1 + P2) / 2
    
    return np.mean(prices), np.std(prices) / np.sqrt(n_sims)

# Paramètres de la Table I
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.10
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 180
n_sims = 5000

# Grille de S/X comme dans le papier
SX_values = np.arange(0.80, 1.25, 0.01)

print("=" * 90)
print("TABLE I — Comparaison MC vs Série (Eq.9) vs B-S")
print(f"σ₀ = {sigma0*100:.0f}%, ξ = {xi}, μ = {mu}, T-t = 180 jours, r = {r}")
print("=" * 90)
print(f"{'S/X':>6} {'Prix BS':>10} {'Prix Eq.9':>10} {'Biais Eq.9%':>12} {'Biais MC%':>12} {'Std Err':>10}")
print("-" * 90)

results = []
for sx in SX_values:
    K = S0 / sx  # S/X = S0/K => K = S0/(S/X)
    
    price_bs = bs_call(S0, K, r, T, sigma0)
    price_eq9 = series_solution_eq9(S0, K, r, T, sigma0, xi)
    price_mc, std_err = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
    
    if price_bs > 1e-6:
        bias_eq9 = (price_eq9 - price_bs) / price_bs * 100
        bias_mc = (price_mc - price_bs) / price_bs * 100
    else:
        bias_eq9 = np.nan
        bias_mc = np.nan
    
    results.append((sx, price_bs, price_eq9, bias_eq9, price_mc, bias_mc, std_err))
    
    bias_eq9_str = f"{bias_eq9:12.2f}" if not np.isnan(bias_eq9) else "      ******"
    bias_mc_str = f"{bias_mc:12.2f}" if not np.isnan(bias_mc) else "      ******"
    
    print(f"{sx:6.2f} {price_bs:10.4f} {price_eq9:10.4f} {bias_eq9_str} {bias_mc_str} {std_err:10.4f}")

# Graphe du biais
fig, ax = plt.subplots(figsize=(12, 6))
sx_arr = np.array([r[0] for r in results])
bias_eq9_arr = np.array([r[3] for r in results])
bias_mc_arr = np.array([r[5] for r in results])

mask_eq9 = ~np.isnan(bias_eq9_arr)
mask_mc = ~np.isnan(bias_mc_arr)

ax.plot(sx_arr[mask_eq9], bias_eq9_arr[mask_eq9], 'b-', linewidth=2, label='Biais Eq. 9 (%)')
ax.plot(sx_arr[mask_mc], bias_mc_arr[mask_mc], 'r--', linewidth=2, label='Biais MC (%)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('S/X', fontsize=13)
ax.set_ylabel('Biais B-S (%)', fontsize=13)
ax.set_title('Table I — Biais de pricing B-S (σ₀=10%, ξ=1, μ=0, T=180j)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.88, 1.24)
ax.set_ylim(-5, 15)
plt.tight_layout()
plt.savefig('exp3_table_I.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure sauvegardée: exp3_table_I.png")
