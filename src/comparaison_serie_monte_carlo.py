"""
Reproduction de la Table I du papier
Comparaison Monte Carlo et solution en série (Eq. 9)
Paramètres : σ₀ = 10%, ξ = 1, μ = 0, T-t = 180 jours, r = 0
"""

import numpy as np
from scipy.stats import norm
import os

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

def series_eq9(S, K, r, T, sigma, xi):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    k = xi**2 * T
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nprime = norm.pdf(d1)
    C0 = bs_call(S, K, r, T, sigma)
    if k < 1e-10:
        return C0
    var_factor = 2.0 * (np.exp(k) - k - 1.0) / k**2 - 1.0
    term1 = 0.5 * (S * np.sqrt(T) * nprime * (d1*d2 - 1) / (4 * sigma**3)) * sigma**4 * var_factor
    skew_factor = (np.exp(3*k) - (9+18*k)*np.exp(k) + (8+24*k+18*k**2+6*k**3)) / (3*k**3)
    term2 = (1.0/6.0) * (S * np.sqrt(T) * nprime * ((d1*d2-3)*(d1*d2-1) - (d1**2+d2**2)) / (8*sigma**5)) * sigma**6 * skew_factor
    return C0 + term1 + term2

def mc_price(S0, K, r, T, V0, xi, n_steps, n_sims):
    dt = T / n_steps
    Z = np.random.randn(n_sims, n_steps)
    log_inc = (-xi**2 / 2) * dt + xi * np.sqrt(dt) * Z
    log_V = np.zeros((n_sims, n_steps + 1))
    log_V[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V[:, i+1] = log_V[:, i] + log_inc[:, i]
    V_bar = np.mean(np.exp(log_V), axis=1)
    prices = bs_call_vec(S0, K, r, T, np.sqrt(V_bar))
    return np.mean(prices), np.std(prices) / np.sqrt(n_sims)

# Paramètres Table I
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.10
V0 = sigma0**2
xi = 1.0
n_steps = 180
n_sims = 10000

SX_values = np.arange(0.75, 1.25, 0.01)

header = f"{'S/X':>5}  {'B-S':>8}  {'Eq. 9':>8}  {'Biais Eq.9 (%)':>15}  {'Biais MC (%)':>13}  {'Std Error (%)':>14}"
sep = "-" * len(header)

lines = []
lines.append("Table I")
lines.append("Comparison of Monte Carlo Procedure and Series Solution")
lines.append(f"Option Parameters: sigma_0 = {sigma0*100:.0f}%, xi = {xi:.0f}, mu = 0, T-t = 180 Days")
lines.append("")
lines.append(header)
lines.append(sep)

print(lines[0])
print(lines[1])
print(lines[2])
print()
print(header)
print(sep)

for sx in SX_values:
    K = S0 / sx
    pbs = bs_call(S0, K, r, T, sigma0)
    peq9 = series_eq9(S0, K, r, T, sigma0, xi)
    pmc, se = mc_price(S0, K, r, T, V0, xi, n_steps, n_sims)

    if pbs > 1e-6:
        bias_eq9 = (peq9 - pbs) / pbs * 100
        bias_mc = (pmc - pbs) / pbs * 100
        se_pct = se / pbs * 100
        bias_eq9_str = f"{bias_eq9:15.2f}"
        bias_mc_str = f"{bias_mc:13.2f}"
        se_str = f"{se_pct:14.2f}"
    else:
        bias_eq9_str = f"{'******':>15}"
        bias_mc_str = f"{'******':>13}"
        se_str = f"{'******':>14}"

    line = f"{sx:5.2f}  {pbs:8.4f}  {peq9:8.4f}  {bias_eq9_str}  {bias_mc_str}  {se_str}"
    print(line)
    lines.append(line)

filepath = os.path.join(output_dir, 'table_I.txt')
with open(filepath, 'w') as f:
    f.write('\n'.join(lines))

print(f"\nSauvegarde: {filepath}")