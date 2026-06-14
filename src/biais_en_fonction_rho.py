"""
Reproduction de la Table II du papier sous forme de graphe
Biais en fonction de S/X pour differents rho
Simulation conjointe de S et V (Methode 2)
Parametres : sigma_0 = 15%, xi = 1, mu = 0, r = 0, T = 180 jours
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time

script_name = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
os.makedirs(output_dir, exist_ok=True)

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_joint(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims):
    """MC vectorise, simulation conjointe de S et V, antithetiques sur u."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(1 - rho**2)

    U = np.random.randn(n_sims, n_steps)
    Vn = np.random.randn(n_sims, n_steps)

    S1 = np.full(n_sims, S0)
    V1 = np.full(n_sims, V0)
    S2 = np.full(n_sims, S0)
    V2 = np.full(n_sims, V0)

    for i in range(n_steps):
        u_i = U[:, i]
        v_i = Vn[:, i]

        S1 = S1 * np.exp((r - V1/2) * dt + np.sqrt(np.maximum(V1, 0)) * sqrt_dt * u_i)
        V1 = np.maximum(V1 * np.exp((mu - xi**2/2) * dt + xi * sqrt_dt * (rho * u_i + sqrt_1_rho2 * v_i)), 1e-12)

        S2 = S2 * np.exp((r - V2/2) * dt + np.sqrt(np.maximum(V2, 0)) * sqrt_dt * (-u_i))
        V2 = np.maximum(V2 * np.exp((mu - xi**2/2) * dt + xi * sqrt_dt * (rho * (-u_i) + sqrt_1_rho2 * v_i)), 1e-12)

    p1 = np.exp(-r * T) * np.maximum(S1 - K, 0)
    p2 = np.exp(-r * T) * np.maximum(S2 - K, 0)
    payoffs = (p1 + p2) / 2
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(n_sims)

# Parametres
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 90
n_sims = 15000

SX = np.linspace(0.88, 1.12, 25)
rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
colors = {-1.0: 'darkblue', -0.5: 'royalblue', 0.0: 'green', 0.5: 'orange', 1.0: 'red'}

fig, ax = plt.subplots(figsize=(10, 6))

for rho in rho_values:
    t0 = time.time()
    bias_pct = []
    for sx in SX:
        K = S0 / sx
        pbs = bs_call(S0, K, r, T, sigma0)
        phw, se = mc_joint(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims)
        if pbs > 1e-6:
            bias_pct.append((phw - pbs) / pbs * 100)
        else:
            bias_pct.append(np.nan)
    ax.plot(SX, bias_pct, 'o-', color=colors[rho], linewidth=2, markersize=4,
            label=f'rho = {rho:.1f}')
    print(f"  rho = {rho:+.1f} termine ({time.time()-t0:.1f}s)")

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('S / X', fontsize=13)
ax.set_ylabel('Biais B-S (%)', fontsize=13)
ax.set_title(r'Biais de pricing pour differents $\rho$ ($\sigma_0$=15%, $\xi$=1, T=180j)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

filepath = os.path.join(output_dir, 'biais_vs_rho.png')
plt.tight_layout()
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegarde: {filepath}")