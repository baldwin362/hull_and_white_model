"""
Expérience 6 — Étude de convergence Monte Carlo
(a) Erreur standard en fonction de N
(b) Biais de discrétisation en fonction de n
(c) Comparaison avec/sans variables antithétiques
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims, antithetic=True):
    """MC Hull-White, retourne (prix moyen, std error, array des prix individuels)."""
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
        
        if antithetic:
            V_anti = np.zeros(n_steps + 1)
            V_anti[0] = V0
            for i in range(n_steps):
                V_anti[i+1] = V_anti[i] * np.exp((mu - xi**2/2) * dt + xi * np.sqrt(dt) * (-z[i]))
            V_bar2 = np.mean(V_anti)
            P2 = bs_call(S0, K, r, T, np.sqrt(max(V_bar2, 1e-12)))
            prices[sim] = (P1 + P2) / 2
        else:
            prices[sim] = P1
    
    return np.mean(prices), np.std(prices) / np.sqrt(n_sims), prices

# Paramètres (option ATM)
S0 = 1.0
K = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0

price_bs = bs_call(S0, K, r, T, sigma0)
print(f"Prix B-S de référence: {price_bs:.6f}")

# =====================================================
# (a) Erreur standard en fonction de N
# =====================================================
print("\n(a) Convergence en N (n_steps=180 fixé)...")
N_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
n_steps_fixed = 180
std_errs_N = []
prices_N = []

for N in N_values:
    t0 = time.time()
    price, se, _ = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps_fixed, N)
    elapsed = time.time() - t0
    std_errs_N.append(se)
    prices_N.append(price)
    bias = (price - price_bs) / price_bs * 100
    print(f"  N={N:6d}: prix={price:.6f}, SE={se:.6f}, biais={bias:.3f}%, temps={elapsed:.1f}s")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.loglog(N_values, std_errs_N, 'bo-', markersize=8, linewidth=2, label='Erreur standard (MC)')
# Droite de référence 1/sqrt(N)
N_ref = np.array(N_values, dtype=float)
ref = std_errs_N[0] * np.sqrt(N_values[0]) / np.sqrt(N_ref)
ax1.loglog(N_values, ref, 'r--', linewidth=1.5, alpha=0.7, label='Pente théorique 1/√N')
ax1.set_xlabel('Nombre de simulations N', fontsize=13)
ax1.set_ylabel('Erreur standard', fontsize=13)
ax1.set_title('(a) Convergence MC en fonction de N\n(option ATM, n=180, ξ=1)', fontsize=13)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('exp6a_convergence_N.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================
# (b) Biais de discrétisation en fonction de n
# =====================================================
print("\n(b) Biais de discrétisation en n (N=10000 fixé)...")
n_values = [10, 25, 50, 100, 180, 252, 500]
N_fixed = 10000
prices_n = []
std_errs_n = []

for n in n_values:
    price, se, _ = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n, N_fixed)
    prices_n.append(price)
    std_errs_n.append(se)
    bias = (price - price_bs) / price_bs * 100
    print(f"  n={n:4d}: prix={price:.6f}, SE={se:.6f}, biais={bias:.3f}%")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(n_values, prices_n, 'bo-', markersize=8, linewidth=2, label='Prix MC')
ax2.axhline(price_bs, color='green', linestyle='--', linewidth=1.5, label=f'Prix B-S = {price_bs:.4f}')
ax2.fill_between(n_values,
                  [p - 2*s for p, s in zip(prices_n, std_errs_n)],
                  [p + 2*s for p, s in zip(prices_n, std_errs_n)],
                  alpha=0.2, color='blue', label='±2 SE')
ax2.set_xlabel('Nombre de pas n', fontsize=13)
ax2.set_ylabel('Prix estimé', fontsize=13)
ax2.set_title('(b) Biais de discrétisation en fonction de n\n(option ATM, N=10000, ξ=1)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp6b_discretisation_n.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================
# (c) Comparaison avec/sans variables antithétiques
# =====================================================
print("\n(c) Comparaison avec/sans variables antithétiques (N=5000, n=180)...")
N_comp = 5000
n_comp = 180
n_repeats = 20

variances_with = []
variances_without = []

for rep in range(n_repeats):
    np.random.seed(rep * 42)
    _, _, prices_with = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_comp, N_comp, antithetic=True)
    variances_with.append(np.var(prices_with))
    
    np.random.seed(rep * 42 + 1000)
    _, _, prices_without = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_comp, N_comp, antithetic=False)
    variances_without.append(np.var(prices_without))

var_with = np.mean(variances_with)
var_without = np.mean(variances_without)
reduction = (1 - var_with / var_without) * 100

print(f"  Variance SANS antithétiques: {var_without:.8f}")
print(f"  Variance AVEC antithétiques: {var_with:.8f}")
print(f"  Réduction de variance: {reduction:.1f}%")
print(f"  Ratio: {var_without/var_with:.2f}x plus efficace avec antithétiques")

fig3, ax3 = plt.subplots(figsize=(8, 5))
bars = ax3.bar(['Sans antithétiques', 'Avec antithétiques'],
               [var_without, var_with],
               color=['#e74c3c', '#2ecc71'], width=0.5)
ax3.set_ylabel('Variance de l\'estimateur', fontsize=13)
ax3.set_title(f'(c) Réduction de variance par variables antithétiques\n'
              f'Réduction: {reduction:.1f}% (facteur {var_without/var_with:.1f}x)', fontsize=13)
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, [var_without, var_with]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
             f'{val:.2e}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('exp6c_antithetic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigures sauvegardées: exp6a_convergence_N.png, exp6b_discretisation_n.png, exp6c_antithetic.png")
