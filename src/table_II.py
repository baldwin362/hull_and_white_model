"""
Reproduction de la Table II du papier
Price Bias as a Percentage of the Black-Scholes Price for Varying
Values of S/X and Correlation rho
Parametres : sigma_0 = 15%, r = 0, xi = 1, mu = 0
"""

import numpy as np
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
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 90
n_sims = 15000

SX_values = [0.90, 0.95, 1.00, 1.05, 1.10]
rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
T_configs = [(90, 90/365), (180, 180/365), (270, 270/365)]

lines = []
lines.append("Table II")
lines.append("Price Bias as a Percentage of the Black-Scholes Price for Varying")
lines.append("Values of S/X and Correlation, rho, between the Volatility and the")
lines.append(f"Stock Price; Option Parameters: sigma_0 = {sigma0*100:.0f}%, r = {r}, xi = {xi:.0f}, and mu = {mu:.0f}")
lines.append("")

print(lines[0])
print(lines[1])
print(lines[2])
print(lines[3])
print()

for T_days, T in T_configs:
    # Header
    sx_header = "".join(f"{sx:>10}" for sx in SX_values)
    block = []
    block.append(f"T = {T_days} Days")
    block.append(f"{'rho':>6}{sx_header}")
    block.append("-" * (6 + 10 * len(SX_values)))

    print(block[0])
    print(block[1])
    print(block[2])
    lines.extend(block)

    for rho in rho_values:
        t0 = time.time()
        bias_line = f"{rho:6.1f}"
        se_line = f"{'':>6}"

        for sx in SX_values:
            K = S0 / sx
            pbs = bs_call(S0, K, r, T, sigma0)
            phw, se = mc_joint(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims)

            if pbs > 1e-6:
                bias_pct = (phw - pbs) / pbs * 100
                se_pct = se / pbs * 100
                bias_line += f"{bias_pct:10.2f}"
                se_line += f"{'(' + f'{se_pct:.2f}' + ')':>10}"
            else:
                bias_line += f"{'******':>10}"
                se_line += f"{'******':>10}"

        elapsed = time.time() - t0
        print(bias_line)
        print(se_line)
        lines.append(bias_line)
        lines.append(se_line)
        print(f"  [rho={rho:+.1f}, {elapsed:.1f}s]")

    print()
    lines.append("")

filepath = os.path.join(output_dir, 'table_II.txt')
with open(filepath, 'w') as f:
    f.write('\n'.join(lines))

print(f"Sauvegarde: {filepath}")