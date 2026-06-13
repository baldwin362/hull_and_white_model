"""
Expérience 9 — Reproduction de la Table III du papier
Volatilités implicites par S/X, ρ et T.
Paramètres : σ₀ = 15%, ξ = 1, μ = 0, r = 0.
"""

import numpy as np
from scipy.stats import norm

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
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

def mc_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims):
    """MC avec simulation conjointe de S et V, antithétiques sur u."""
    dt = T / n_steps
    payoffs = np.zeros(n_sims)
    payoffs_sq = np.zeros(n_sims)
    
    for sim in range(n_sims):
        u = np.random.randn(n_steps)
        v = np.random.randn(n_steps)
        
        S1, V1 = S0, V0
        S2, V2 = S0, V0
        
        for i in range(n_steps):
            # Directe
            S1 = S1 * np.exp((r - V1/2) * dt + np.sqrt(max(V1,0)) * np.sqrt(dt) * u[i])
            V1 = V1 * np.exp((mu - xi**2/2) * dt + xi * np.sqrt(dt) * (rho * u[i] + np.sqrt(1 - rho**2) * v[i]))
            V1 = max(V1, 1e-12)
            # Antithétique
            S2 = S2 * np.exp((r - V2/2) * dt + np.sqrt(max(V2,0)) * np.sqrt(dt) * (-u[i]))
            V2 = V2 * np.exp((mu - xi**2/2) * dt + xi * np.sqrt(dt) * (rho * (-u[i]) + np.sqrt(1 - rho**2) * v[i]))
            V2 = max(V2, 1e-12)
        
        p1 = np.exp(-r * T) * max(S1 - K, 0)
        p2 = np.exp(-r * T) * max(S2 - K, 0)
        payoffs[sim] = (p1 + p2) / 2
    
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(n_sims)

def mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    """MC ρ=0 : simule V seul."""
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
    return np.mean(prices), np.std(prices) / np.sqrt(n_sims)

# Paramètres
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 180
n_sims = 10000

SX_values = [0.90, 0.95, 1.00, 1.05, 1.10]
rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
T_values = [(90, 90/365), (180, 180/365), (270, 270/365)]

print("=" * 90)
print("TABLE III — Volatilités implicites calculées par inversion de Black-Scholes")
print(f"Volatilité attendue : σ₀ = {sigma0*100:.0f}%")
print(f"Paramètres : ξ = {xi}, μ = {mu}, r = {r}")
print("=" * 90)

for T_days, T in T_values:
    print(f"\n{'='*80}")
    print(f"T = {T_days} jours")
    print(f"{'ρ':>6}", end="")
    for sx in SX_values:
        print(f"{'S/X='+str(sx):>14}", end="")
    print()
    print("-" * 80)
    
    for rho in rho_values:
        print(f"{rho:6.1f}", end="")
        for sx in SX_values:
            K = S0 / sx
            
            if abs(rho) < 1e-10:
                price, se = mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
            else:
                price, se = mc_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims)
            
            sig_imp = implied_vol_bisection(S0, K, r, T, price)
            # Erreur standard sur la vol implicite (approximation via vega)
            if not np.isnan(sig_imp):
                vega = S0 * np.sqrt(T) * norm.pdf(
                    (np.log(S0/K) + (r + sig_imp**2/2)*T) / (sig_imp * np.sqrt(T)))
                if vega > 1e-10:
                    se_imp = se / vega
                else:
                    se_imp = np.nan
                print(f"  {sig_imp*100:6.2f}({se_imp*100:.2f})" if not np.isnan(se_imp) else f"  {sig_imp*100:6.2f}(  ??)", end="")
            else:
                print(f"       ******", end="")
        print()
        
print("\n\nTable III reproduite avec succès.")
print("Comparer avec le papier (Table III, p.297):")
print("  Pour ρ=0, T=90j, ATM: papier donne 14.86%, nous devons obtenir ~14.8-14.9%")
print("  L'effet maturité: pour ρ=0, σ_imp ATM décroît quand T augmente")
