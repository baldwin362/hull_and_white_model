"""
Reproduction de la Figure 1 du papier
Pricing Bias When μ = 0, r = 0, σ_t = 15%, ξ = 1, T-t = 180 Days
Prix B-S vs prix vrai (Eq. 9) avec biais amplifié 25 fois.
"""

import numpy as np
import matplotlib.pyplot as plt
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

# Paramètres Figure 1
S0 = 1.0
r = 0.0
T = 180 / 365
sigma0 = 0.15
xi = 1.0

SX = np.linspace(0.75, 1.25, 300)

prices_bs = []
prices_true = []
for sx in SX:
    K = S0 / sx
    prices_bs.append(bs_call(S0, K, r, T, sigma0))
    prices_true.append(series_eq9(S0, K, r, T, sigma0, xi))

prices_bs = np.array(prices_bs)
prices_true = np.array(prices_true)
bias = prices_true - prices_bs
prices_exaggerated = prices_bs + 25 * bias

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(SX, prices_bs, 'k-', linewidth=2, label='Black-Scholes Price')
ax.plot(SX, prices_exaggerated, 'k--', linewidth=2, label='True Price (Equation 9)\n(Bias exaggerated 25-fold)')

# Trouver les croisements (changements de signe du biais)
crossings = []
for i in range(len(bias) - 1):
    if bias[i] * bias[i+1] < 0:
        sx_cross = SX[i] - bias[i] * (SX[i+1] - SX[i]) / (bias[i+1] - bias[i])
        crossings.append(sx_cross)

for sx_cross in crossings:
    ax.axvline(sx_cross, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate(f'S/X = {sx_cross:.2f}', xy=(sx_cross, 0.24), fontsize=10,
                color='red', ha='center')

ax.set_xlabel('S/X', fontsize=14)
ax.set_ylabel('OPTION PRICE', fontsize=14)
ax.set_title(r'Pricing Bias When $\mu = 0$, $r = 0$, $\sigma_t = 15\%$, $\xi = 1$, $T - t = 180$ Days', fontsize=13)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.75, 1.25)
ax.set_ylim(-0.01, 0.26)

filepath = os.path.join(output_dir, 'figure_1.png')
plt.tight_layout()
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.show()
print(f"Sauvegarde: {filepath}")