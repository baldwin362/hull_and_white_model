"""
Expérience 1 — Effet Jensen
Trace C_BS en fonction de V_bar pour deep ITM, ITM, ATM, OTM et deep OTM.
Montre la concavité (ATM) vs convexité (ITM/OTM) et le biais de Jensen
mesuré par Monte Carlo.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
os.makedirs('exp_1', exist_ok=True)

def bs_call(S, K, r, T, sigma):
    """Prix Black-Scholes d'un call européen."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_vec(S, K, r, T, sigma_vec):
    """Prix B-S vectorisé."""
    sigma_vec = np.maximum(sigma_vec, 1e-12)
    d1 = (np.log(S / K) + (r + sigma_vec**2 / 2) * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_expected_cbs(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    """Calcule E[C_BS(sqrt(V_bar))] par MC vectorisé."""
    dt = T / n_steps
    Z = np.random.randn(n_sims, n_steps)
    log_inc = (mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * Z
    log_V = np.zeros((n_sims, n_steps + 1))
    log_V[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V[:, i+1] = log_V[:, i] + log_inc[:, i]
    V_bar = np.mean(np.exp(log_V), axis=1)
    prices = bs_call_vec(S0, K, r, T, np.sqrt(V_bar))
    return np.mean(prices), V_bar

# Paramètres
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 90
n_sims = 20000

# Grille de V_bar pour la courbe théorique
V_bar_grid = np.linspace(0.001, 0.10, 500)

# Cinq options
strikes_and_labels = [
    (0.80 * S0, 'Deep ITM (K/S₀ = 0.80)', 'deep_itm'),
    (0.90 * S0, 'ITM (K/S₀ = 0.90)', 'itm'),
    (S0,        'ATM (K/S₀ = 1.00)', 'atm'),
    (1.10 * S0, 'OTM (K/S₀ = 1.10)', 'otm'),
    (1.20 * S0, 'Deep OTM (K/S₀ = 1.20)', 'deep_otm'),
]

for K, label, filename in strikes_and_labels:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Courbe C_BS(V_bar)
    C = np.array([bs_call(S0, K, r, T, np.sqrt(v)) for v in V_bar_grid])
    ax.plot(V_bar_grid, C, 'b-', linewidth=2, label='C_BS(√V̄)')
    
    # Point rouge : C(E[V_bar]) = prix B-S naïf
    C_at_V0 = bs_call(S0, K, r, T, sigma0)
    ax.axvline(V0, color='red', linestyle='--', alpha=0.5)
    ax.plot(V0, C_at_V0, 'ro', markersize=10, zorder=5,
            label=f'C(E[V̄]) = {C_at_V0:.4f}  (prix B-S)')
    
    # Triangle vert : E[C(V_bar)] = vrai prix Hull-White par MC
    E_C, V_bar_samples = mc_expected_cbs(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
    ax.plot(V0, E_C, 'g^', markersize=12, zorder=5,
            label=f'E[C(V̄)] = {E_C:.4f}  (prix HW)')
    
    # Annotation du biais
    if C_at_V0 > 1e-6:
        biais_pct = (E_C - C_at_V0) / C_at_V0 * 100
        signe = "surpaye" if biais_pct < 0 else "sous-paye"
        ax.annotate(f'Biais = {biais_pct:+.2f}%\nB-S {signe}',
                    xy=(V0 + 0.003, (E_C + C_at_V0) / 2),
                    fontsize=11, color='darkgreen', fontweight='bold')
    
    ax.set_xlabel('V̄ (variance moyenne)', fontsize=12)
    ax.set_ylabel('C_BS(V̄)', fontsize=12)
    ax.set_title(f'Effet Jensen : {label}\n(σ₀=15%, ξ=1, r=0, T=180j)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f'exp_1/{filename}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Sauvegardé: {path}")