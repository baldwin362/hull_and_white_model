"""
Expérience 4 — Reproduction de la Figure 1 du papier
Prix B-S vs prix vrai (Eq. 9) avec biais amplifié 25 fois.
Paramètres : σ₀ = 15%, ξ = 1, μ = 0, r = 0, T-t = 180 jours.
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
    """Solution en série (Eq. 9), cas μ=0, ρ=0."""
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    
    k = xi**2 * T
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nprime_d1 = norm.pdf(d1)
    
    C0 = bs_call(S, K, r, T, sigma)
    
    if k < 1e-10:
        return C0
    
    var_factor = 2.0 * (np.exp(k) - k - 1.0) / k**2 - 1.0
    C_second = S * np.sqrt(T) * nprime_d1 * (d1 * d2 - 1) / (4 * sigma**3)
    term1 = 0.5 * C_second * sigma**4 * var_factor
    
    skew_factor = (np.exp(3*k) - (9 + 18*k) * np.exp(k) + (8 + 24*k + 18*k**2 + 6*k**3)) / (3 * k**3)
    C_third = S * np.sqrt(T) * nprime_d1 * ((d1*d2-3)*(d1*d2-1) - (d1**2 + d2**2)) / (8 * sigma**5)
    term2 = (1.0/6.0) * C_third * sigma**6 * skew_factor
    
    return C0 + term1 + term2

# Paramètres de la Figure 1
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.15
xi = 1.0

# Grille de S/X
SX = np.linspace(0.75, 1.25, 300)

prices_bs = []
prices_true = []
for sx in SX:
    K = S0 / sx
    pbs = bs_call(S0, K, r, T, sigma0)
    ptrue = series_solution_eq9(S0, K, r, T, sigma0, xi)
    prices_bs.append(pbs)
    prices_true.append(ptrue)

prices_bs = np.array(prices_bs)
prices_true = np.array(prices_true)
bias = prices_true - prices_bs

# Construire le "true price" avec biais amplifié 25x pour la visualisation
prices_exaggerated = prices_bs + 25 * bias

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(SX, prices_bs, 'b-', linewidth=2, label='Prix Black-Scholes')
ax.plot(SX, prices_exaggerated, 'r--', linewidth=2, label='Prix vrai (Eq. 9)\n(biais amplifié ×25)')
ax.set_xlabel('S / X', fontsize=14)
ax.set_ylabel('Prix de l\'option', fontsize=14)
ax.set_title('Figure 1 — Biais de pricing (μ=0, r=0, σ₀=15%, ξ=1, T=180j)\n'
             'Le biais est amplifié 25 fois pour être visible', fontsize=13)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.75, 1.25)
ax.set_ylim(-0.01, 0.26)
plt.tight_layout()
plt.savefig('exp4_figure1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: exp4_figure1.png")

# Tracer aussi le biais en % séparément
fig2, ax2 = plt.subplots(figsize=(10, 5))
mask = prices_bs > 1e-6
bias_pct = np.full_like(bias, np.nan)
bias_pct[mask] = bias[mask] / prices_bs[mask] * 100
ax2.plot(SX[mask], bias_pct[mask], 'b-', linewidth=2)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('S / X', fontsize=14)
ax2.set_ylabel('Biais B-S (%)', fontsize=14)
ax2.set_title('Biais en % du prix B-S (σ₀=15%, ξ=1, T=180j)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.88, 1.25)
ax2.set_ylim(-5, 10)
plt.tight_layout()
plt.savefig('exp4_figure1_bias_pct.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: exp4_figure1_bias_pct.png")
