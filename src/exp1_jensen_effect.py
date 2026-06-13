"""
Expérience 1 — Effet Jensen
Trace C_BS en fonction de V_bar pour ATM, OTM et deep OTM.
Montre la concavité (ATM) vs convexité (OTM) qui explique le biais.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call(S, K, r, T, sigma):
    """Prix Black-Scholes d'un call européen."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Paramètres
S0 = 1.0
r = 0.0
T = 180 / 365  # 180 jours
sigma0 = 0.15
V0 = sigma0**2

# Grille de V_bar
V_bar = np.linspace(0.001, 0.10, 500)

# Trois options: ATM, OTM, deep OTM
# ATM: K = S0 * exp(-rT) = S0 quand r=0
K_atm = S0
K_otm = 1.10 * S0    # S/X = 0.91
K_deep_otm = 1.20 * S0  # S/X = 0.83

# Calcul de C_BS(V_bar) pour chaque strike
C_atm = np.array([bs_call(S0, K_atm, r, T, np.sqrt(v)) for v in V_bar])
C_otm = np.array([bs_call(S0, K_otm, r, T, np.sqrt(v)) for v in V_bar])
C_deep = np.array([bs_call(S0, K_deep_otm, r, T, np.sqrt(v)) for v in V_bar])

# Tracer
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, C, K, label in zip(axes,
                            [C_atm, C_otm, C_deep],
                            [K_atm, K_otm, K_deep_otm],
                            ['ATM (K/S₀ = 1.00)', 'OTM (K/S₀ = 1.10)', 'Deep OTM (K/S₀ = 1.20)']):
    ax.plot(V_bar, C, 'b-', linewidth=2)
    # Marquer E[V_bar] = V0
    C_at_V0 = bs_call(S0, K, r, T, sigma0)
    ax.axvline(V0, color='red', linestyle='--', alpha=0.7, label=f'E[V̄] = σ₀² = {V0:.4f}')
    ax.plot(V0, C_at_V0, 'ro', markersize=8)
    ax.set_xlabel('V̄ (variance moyenne)', fontsize=12)
    ax.set_ylabel('C_BS(V̄)', fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effet Jensen : concavité/convexité de C_BS en fonction de V̄\n'
             '(σ₀=15%, r=0, T=180j)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('exp1_jensen_effect.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: exp1_jensen_effect.png")
