"""
Expérience 10 — Effet maturité avec r ≠ 0 (Figure 4 du papier)
Paramètres : r = 10%, σ₀ = 15%, ξ = 1, μ = 0.
T-t = 45, 90, 135 jours.
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

def mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    """MC ρ=0 : simule V seul, avec antithétiques."""
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

# Paramètres de la Figure 4
S0 = 1.0
r = 0.10  # <-- r non nul !
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 180
n_sims = 20000

T_configs = [(45, 45/365), (90, 90/365), (135, 135/365)]
SX = np.linspace(0.88, 1.12, 30)

fig, ax = plt.subplots(figsize=(10, 6))
colors = {'45': 'blue', '90': 'green', '135': 'red'}
styles = {'45': '-', '90': '--', '135': ':'}

for T_days, T in T_configs:
    print(f"Calcul T = {T_days} jours...")
    bias_pct_list = []
    
    for j, sx in enumerate(SX):
        K = S0 / sx
        price_bs = bs_call(S0, K, r, T, sigma0)
        price_hw = mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
        
        if price_bs > 1e-6:
            bias_pct = (price_hw - price_bs) / price_bs * 100
        else:
            bias_pct = np.nan
        bias_pct_list.append(bias_pct)
        
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(SX)} strikes calculés")
    
    label = f'T−t = {T_days} jours'
    ax.plot(SX, bias_pct_list,
            linestyle=styles[str(T_days)],
            color=colors[str(T_days)],
            linewidth=2, label=label)

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('S / X', fontsize=13)
ax.set_ylabel('Biais B-S (%)', fontsize=13)
ax.set_title('Figure 4 — Effet de T avec r = 10%\n'
             '(σ₀=15%, ξ=1, μ=0)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 5)
plt.tight_layout()
plt.savefig('exp10_figure4_maturity.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure sauvegardée: exp10_figure4_maturity.png")

# Commentaire
print("\nNOTE: Avec r=10%, le point ATM au sens du papier est S = X*exp(-r(T-t)),")
print("donc le centre du biais se déplace vers la gauche quand T augmente.")
print("C'est exactement ce que la Figure 4 du papier montre.")
