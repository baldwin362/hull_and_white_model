"""
Expérience 7 — Reproduction de la Table II du papier
Biais pour ρ ≠ 0 (simulation conjointe de S et V, Méthode 2).
Paramètres : σ₀ = 15%, ξ = 1, μ = 0, r = 0.
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

def mc_hull_white_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims):
    """
    Méthode 2 du papier : simulation conjointe de S et V avec ρ ≠ 0.
    Utilise variables antithétiques (4 combinaisons) + variable de contrôle.
    """
    dt = T / n_steps
    
    bias_estimates = np.zeros(n_sims)
    
    for sim in range(n_sims):
        u = np.random.randn(n_steps)
        v = np.random.randn(n_steps)
        
        payoffs = []
        # 4 combinaisons antithétiques : (u,v), (-u,v), (u,-v), (-u,-v)
        for su, sv in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            S = np.zeros(n_steps + 1)
            V = np.zeros(n_steps + 1)
            S[0] = S0
            V[0] = V0
            for i in range(n_steps):
                ui = su * u[i]
                vi = sv * v[i]
                S[i+1] = S[i] * np.exp((r - V[i]/2) * dt + np.sqrt(max(V[i], 0)) * np.sqrt(dt) * ui)
                V[i+1] = V[i] * np.exp((mu - xi**2/2) * dt + xi * np.sqrt(dt) * (rho * ui + np.sqrt(1 - rho**2) * vi))
                V[i+1] = max(V[i+1], 1e-12)
            payoffs.append(np.exp(-r * T) * max(S[-1] - K, 0))
        
        # Variable de contrôle : B-S simulé sur les mêmes u, avec V fixe
        bs_payoffs = []
        for su in [1, -1]:
            S_bs = np.zeros(n_steps + 1)
            S_bs[0] = S0
            sigma_fixed = np.sqrt(V0)
            for i in range(n_steps):
                ui = su * u[i]
                S_bs[i+1] = S_bs[i] * np.exp((r - V0/2) * dt + sigma_fixed * np.sqrt(dt) * ui)
            bs_payoffs.append(np.exp(-r * T) * max(S_bs[-1] - K, 0))
        
        # Estimateur du biais avec variable de contrôle
        est1 = (payoffs[0] + payoffs[2] - 2 * bs_payoffs[0]) / 2
        est2 = (payoffs[1] + payoffs[3] - 2 * bs_payoffs[1]) / 2
        bias_estimates[sim] = (est1 + est2) / 2
    
    price_bs = bs_call(S0, K, r, T, np.sqrt(V0))
    mean_bias = np.mean(bias_estimates)
    std_err_bias = np.std(bias_estimates) / np.sqrt(n_sims)
    price_hw = price_bs + mean_bias
    
    return price_hw, mean_bias, std_err_bias

# Paramètres de la Table II
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 180
n_sims = 5000

SX_values = [0.90, 0.95, 1.00, 1.05, 1.10]
rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
T_values = [90/365, 180/365, 270/365]

print("=" * 90)
print("TABLE II — Biais en % du prix B-S pour différents ρ, S/X, T")
print(f"σ₀ = {sigma0*100:.0f}%, ξ = {xi}, μ = {mu}, r = {r}")
print("=" * 90)

for T in T_values:
    T_days = int(T * 365)
    print(f"\n{'='*90}")
    print(f"T = {T_days} jours")
    print(f"{'ρ':>6}", end="")
    for sx in SX_values:
        print(f"{'S/X='+str(sx):>14}", end="")
    print()
    print("-" * 90)
    
    for rho in rho_values:
        print(f"{rho:6.1f}", end="")
        for sx in SX_values:
            K = S0 / sx
            price_bs = bs_call(S0, K, r, T, sigma0)
            price_hw, bias, se = mc_hull_white_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims)
            
            if price_bs > 1e-6:
                bias_pct = bias / price_bs * 100
                se_pct = se / price_bs * 100
                print(f"  {bias_pct:7.2f}({se_pct:.2f})", end="")
            else:
                print(f"       ******", end="")
        print()

print("\n\nTable II reproduite avec succès.")
