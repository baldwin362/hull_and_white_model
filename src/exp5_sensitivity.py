"""
Expérience 5 — Sensibilité à σ₀ et ξ (Figures 2 et 3 du papier)
(a) Biais en fonction de S/X pour σ₀ = 10%, 15%, 20% (ξ=1 fixé)
(b) Biais en fonction de S/X pour ξ = 1, 2, 3 (σ₀=15% fixé)
Pour ξ=2,3 la série ne converge plus bien → on utilise MC.
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
    skew_factor = (np.exp(3*k) - (9 + 18*k)*np.exp(k) + (8 + 24*k + 18*k**2 + 6*k**3)) / (3 * k**3)
    C_third = S * np.sqrt(T) * nprime_d1 * ((d1*d2-3)*(d1*d2-1) - (d1**2+d2**2)) / (8 * sigma**5)
    term2 = (1.0/6.0) * C_third * sigma**6 * skew_factor
    return C0 + term1 + term2

def mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    dt = T / n_steps
    prices = np.zeros(n_sims)
    for sim in range(n_sims):
        z = np.random.randn(n_steps)
        V = np.zeros(n_steps + 1)
        V[0] = V0
        for i in range(n_steps):
            V[i+1] = V[i] * np.exp((mu - xi**2/2) * dt + xi * np.sqrt(dt) * z[i])
        V_bar1 = np.mean(V)
        P1 = bs_call(S0, K, r, T, np.sqrt(max(V_bar1, 1e-12)))
        V_anti = np.zeros(n_steps + 1)
        V_anti[0] = V0
        for i in range(n_steps):
            V_anti[i+1] = V_anti[i] * np.exp((mu - xi**2/2) * dt + xi * np.sqrt(dt) * (-z[i]))
        V_bar2 = np.mean(V_anti)
        P2 = bs_call(S0, K, r, T, np.sqrt(max(V_bar2, 1e-12)))
        prices[sim] = (P1 + P2) / 2
    return np.mean(prices)

# Paramètres communs
S0 = 1.0
r = 0.0
T = 180 / 365
mu = 0.0
n_steps = 180

SX = np.linspace(0.88, 1.12, 50)

# =====================================================
# Figure 2 : Effet de σ₀ (ξ = 1 fixé, série converge)
# =====================================================
print("Calcul Figure 2 — Effet de σ₀...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

for sigma0, style in [(0.10, '-'), (0.15, '--'), (0.20, ':')]:
    xi = 1.0
    bias_pct = []
    for sx in SX:
        K = S0 / sx
        pbs = bs_call(S0, K, r, T, sigma0)
        ptrue = series_solution_eq9(S0, K, r, T, sigma0, xi)
        if pbs > 1e-6:
            bias_pct.append((ptrue - pbs) / pbs * 100)
        else:
            bias_pct.append(np.nan)
    ax1.plot(SX, bias_pct, linestyle=style, linewidth=2, label=f'σ₀ = {sigma0*100:.0f}%')

ax1.axhline(0, color='black', linewidth=0.5)
ax1.set_xlabel('S / X', fontsize=13)
ax1.set_ylabel('Biais B-S (%)', fontsize=13)
ax1.set_title('Figure 2 — Effet de σ₀ (ξ=1, μ=0, r=0, T=180j)', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 5)
plt.tight_layout()
plt.savefig('exp5_figure2_sigma.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: exp5_figure2_sigma.png")

# =====================================================
# Figure 3 : Effet de ξ (σ₀ = 15% fixé)
# Pour ξ=1 on utilise la série, pour ξ=2,3 le MC
# =====================================================
print("\nCalcul Figure 3 — Effet de ξ (MC pour ξ=2,3, peut prendre quelques minutes)...")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sigma0 = 0.15
V0 = sigma0**2
n_sims = 10000

for xi, style in [(1.0, '-'), (2.0, '--'), (3.0, ':')]:
    bias_pct = []
    for j, sx in enumerate(SX):
        K = S0 / sx
        pbs = bs_call(S0, K, r, T, sigma0)
        
        if xi <= 1.0:
            ptrue = series_solution_eq9(S0, K, r, T, sigma0, xi)
        else:
            ptrue = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
        
        if pbs > 1e-6:
            bias_pct.append((ptrue - pbs) / pbs * 100)
        else:
            bias_pct.append(np.nan)
        
        if (j + 1) % 10 == 0:
            print(f"  ξ={xi}, {j+1}/{len(SX)} strikes calculés")
    
    ax2.plot(SX, bias_pct, linestyle=style, linewidth=2, label=f'ξ = {xi:.0f}')

ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('S / X', fontsize=13)
ax2.set_ylabel('Biais B-S (%)', fontsize=13)
ax2.set_title('Figure 3 — Effet de ξ (σ₀=15%, μ=0, r=0, T=180j)', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-25, 5)
plt.tight_layout()
plt.savefig('exp5_figure3_xi.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: exp5_figure3_xi.png")
