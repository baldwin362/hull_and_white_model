"""
Reproduction de la Figure 3 du papier
Effect of Varying xi When mu = 0, r = 0, sigma_t = 15%, T-t = 180 Days
Biais en fonction de S/X pour xi = 1, 2, 3
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

def bs_call_vec(S, K, r, T, sigma_vec):
    sigma_vec = np.maximum(sigma_vec, 1e-12)
    d1 = (np.log(S / K) + (r + sigma_vec**2 / 2) * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_rho0(S0, K, r, T, V0, xi, n_steps, n_sims):
    """MC vectorise, rho=0, mu=0, avec antithetiques."""
    dt = T / n_steps
    Z = np.random.randn(n_sims, n_steps)

    log_inc = (-xi**2 / 2) * dt + xi * np.sqrt(dt) * Z
    log_V = np.zeros((n_sims, n_steps + 1))
    log_V[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V[:, i+1] = log_V[:, i] + log_inc[:, i]
    V_bar1 = np.mean(np.exp(log_V), axis=1)
    P1 = bs_call_vec(S0, K, r, T, np.sqrt(V_bar1))

    log_inc_anti = (-xi**2 / 2) * dt + xi * np.sqrt(dt) * (-Z)
    log_V_anti = np.zeros((n_sims, n_steps + 1))
    log_V_anti[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V_anti[:, i+1] = log_V_anti[:, i] + log_inc_anti[:, i]
    V_bar2 = np.mean(np.exp(log_V_anti), axis=1)
    P2 = bs_call_vec(S0, K, r, T, np.sqrt(V_bar2))

    return np.mean((P1 + P2) / 2)

# Parametres
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.15
V0 = sigma0**2
n_steps = 90
n_sims = 50000

SX = np.linspace(0.88, 1.12, 40)

fig, ax = plt.subplots(figsize=(10, 6))

for xi, style in [(1.0, '-'), (2.0, '-.'), (3.0, ':')]:
    t0 = time.time()
    bias_pct = []
    for sx in SX:
        K = S0 / sx
        pbs = bs_call(S0, K, r, T, sigma0)
        phw = mc_rho0(S0, K, r, T, V0, xi, n_steps, n_sims)
        if pbs > 1e-6:
            bias_pct.append((phw - pbs) / pbs * 100)
        else:
            bias_pct.append(np.nan)
    ax.plot(SX, bias_pct, linestyle=style, color='black', linewidth=2,
            label=r'$\xi$ = ' + f'{xi:.0f}')
    print(f"  xi = {xi:.0f} termine ({time.time()-t0:.1f}s)")

ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(-12.5, color='black', linewidth=0.5, alpha=0.3)
ax.axvline(1.0, color='black', linewidth=0.5)
ax.set_xlabel('S/X', fontsize=14)
ax.set_ylabel('PRICE BIAS (%)', fontsize=14)
ax.set_title(r'Effect of Varying $\xi$ When $\mu = 0$, $r = 0$, $\sigma_t = 15\%$, $T - t = 180$ Days', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.88, 1.12)
ax.set_ylim(-25, 5)

filepath = os.path.join(output_dir, 'figure_3.png')
plt.tight_layout()
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegarde: {filepath}")