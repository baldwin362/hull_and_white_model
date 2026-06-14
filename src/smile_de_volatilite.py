"""
Smiles de volatilite implicite dans le modele de Hull & White
(a) Smile de reference rho=0, xi=1, T=180j
(b) Effet de xi : xi = 0.5, 1, 2
(c) Effet de rho : rho = -1, -0.5, 0, 0.5, 1
(d) Effet de T : T = 90, 180, 270 jours
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

def implied_vol_bisection(S, K, r, T, price, tol=1e-8, max_iter=300):
    lower_bound = max(S - K * np.exp(-r * T), 0.0)
    if price <= lower_bound + 1e-10 or price >= S - 1e-10:
        return np.nan
    sig_low, sig_high = 1e-6, 5.0
    for _ in range(max_iter):
        sig_mid = (sig_low + sig_high) / 2
        p = bs_call(S, K, r, T, sig_mid)
        if abs(p - price) < tol:
            return sig_mid
        if p < price:
            sig_low = sig_mid
        else:
            sig_high = sig_mid
    return sig_mid

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

def mc_joint(S0, K, r, T, V0, xi, rho, n_steps, n_sims):
    """MC vectorise, simulation conjointe S et V, antithetiques sur u."""
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
        V1 = np.maximum(V1 * np.exp((-xi**2/2) * dt + xi * sqrt_dt * (rho * u_i + sqrt_1_rho2 * v_i)), 1e-12)
        S2 = S2 * np.exp((r - V2/2) * dt + np.sqrt(np.maximum(V2, 0)) * sqrt_dt * (-u_i))
        V2 = np.maximum(V2 * np.exp((-xi**2/2) * dt + xi * sqrt_dt * (rho * (-u_i) + sqrt_1_rho2 * v_i)), 1e-12)
    p1 = np.exp(-r * T) * np.maximum(S1 - K, 0)
    p2 = np.exp(-r * T) * np.maximum(S2 - K, 0)
    return np.mean((p1 + p2) / 2)

def compute_smile(rho, xi, T, S0, r, V0, n_steps, n_sims, moneyness):
    """Calcule le smile pour un jeu de parametres."""
    sigma_imps = []
    for sx in moneyness:
        K = S0 / sx
        if abs(rho) < 1e-10:
            price = mc_rho0(S0, K, r, T, V0, xi, n_steps, n_sims)
        else:
            price = mc_joint(S0, K, r, T, V0, xi, rho, n_steps, n_sims)
        sig = implied_vol_bisection(S0, K, r, T, price)
        sigma_imps.append(sig * 100 if not np.isnan(sig) else np.nan)
    return sigma_imps

# Parametres
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
n_steps = 90
n_sims = 50000

moneyness = np.linspace(0.88, 1.12, 20)

# =====================================================
# (a) Smile de reference
# =====================================================
print("(a) Smile de reference...")
t0 = time.time()
smile_ref = compute_smile(0.0, 1.0, 180/365, S0, r, V0, n_steps, n_sims, moneyness)
print(f"  termine ({time.time()-t0:.1f}s)")

fig, ax = plt.subplots(figsize=(10, 6))
valid = [not np.isnan(s) for s in smile_ref]
ax.plot(moneyness[valid], np.array(smile_ref)[valid], 'bo-', linewidth=2, markersize=5)
ax.axhline(sigma0*100, color='red', linestyle='--', alpha=0.7, label=r'$\sigma_0$ = 15%')
ax.set_xlabel('S / X', fontsize=14)
ax.set_ylabel(r'$\sigma_{imp}$ (%)', fontsize=14)
ax.set_title(r'Smile de volatilite implicite ($\rho=0$, $\xi=1$, $T=180j$)', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'smile_reference.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sauvegarde: smile_reference.png")

# =====================================================
# (b) Effet de xi
# =====================================================
print("\n(b) Effet de xi...")
fig, ax = plt.subplots(figsize=(10, 6))
for xi_val, color, marker in [(0.5, 'green', 's'), (1.0, 'blue', 'o'), (2.0, 'red', '^')]:
    t0 = time.time()
    smile = compute_smile(0.0, xi_val, 180/365, S0, r, V0, n_steps, n_sims, moneyness)
    valid = [not np.isnan(s) for s in smile]
    ax.plot(moneyness[valid], np.array(smile)[valid], f'{marker}-', color=color,
            linewidth=2, markersize=5, label=r'$\xi$ = ' + f'{xi_val}')
    print(f"  xi={xi_val} termine ({time.time()-t0:.1f}s)")
ax.axhline(sigma0*100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('S / X', fontsize=14)
ax.set_ylabel(r'$\sigma_{imp}$ (%)', fontsize=14)
ax.set_title(r'Effet de $\xi$ sur le smile ($\rho=0$, $T=180j$)', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'smile_effet_xi.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sauvegarde: smile_effet_xi.png")

# =====================================================
# (c) Effet de rho
# =====================================================
print("\n(c) Effet de rho...")
fig, ax = plt.subplots(figsize=(10, 6))
rho_configs = [
    (-1.0, 'darkblue', 'v'),
    (-0.5, 'royalblue', '<'),
    (0.0, 'green', 'o'),
    (0.5, 'orange', '>'),
    (1.0, 'red', '^'),
]
for rho_val, color, marker in rho_configs:
    t0 = time.time()
    smile = compute_smile(rho_val, 1.0, 180/365, S0, r, V0, n_steps, n_sims, moneyness)
    valid = [not np.isnan(s) for s in smile]
    ax.plot(moneyness[valid], np.array(smile)[valid], f'{marker}-', color=color,
            linewidth=2, markersize=5, label=r'$\rho$ = ' + f'{rho_val}')
    print(f"  rho={rho_val:+.1f} termine ({time.time()-t0:.1f}s)")
ax.axhline(sigma0*100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('S / X', fontsize=14)
ax.set_ylabel(r'$\sigma_{imp}$ (%)', fontsize=14)
ax.set_title(r'Effet de $\rho$ sur le smile ($\xi=1$, $T=180j$)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'smile_effet_rho.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sauvegarde: smile_effet_rho.png")

# =====================================================
# (d) Effet de T
# =====================================================
print("\n(d) Effet de T...")
fig, ax = plt.subplots(figsize=(10, 6))
for T_days, color, marker in [(90, 'blue', 's'), (180, 'green', 'o'), (270, 'red', '^')]:
    t0 = time.time()
    smile = compute_smile(0.0, 1.0, T_days/365, S0, r, V0, n_steps, n_sims, moneyness)
    valid = [not np.isnan(s) for s in smile]
    ax.plot(moneyness[valid], np.array(smile)[valid], f'{marker}-', color=color,
            linewidth=2, markersize=5, label=f'T = {T_days} jours')
    print(f"  T={T_days}j termine ({time.time()-t0:.1f}s)")
ax.axhline(sigma0*100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('S / X', fontsize=14)
ax.set_ylabel(r'$\sigma_{imp}$ (%)', fontsize=14)
ax.set_title(r'Effet de $T$ sur le smile ($\rho=0$, $\xi=1$)', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'smile_effet_T.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Sauvegarde: smile_effet_T.png")

print(f"\nToutes les figures sauvegardees dans {output_dir}/")