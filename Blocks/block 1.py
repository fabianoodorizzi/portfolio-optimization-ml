# 📌 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import seaborn as sns

# 🔹 Step (a) – Synthetic Data Generation (if no real data)
np.random.seed(42)
dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='M')
assets = ["MSCI_USA", "MSCI_Europe", "MSCI_China", "MSCI_Japan", "MSCI_Brazil"]
n_assets = len(assets)

# Simulate prices
prices = pd.DataFrame(
    np.cumprod(1 + 0.005 * np.random.randn(len(dates), n_assets), axis=0) * 100,
    index=dates, columns=assets
)

# 🔹 Step (a) – Log-returns
returns = np.log(prices / prices.shift(1)).dropna()

# 🔹 Step (b) – Rolling window setup
window = 60  # 60 months = 5 years
step = 1
rolling_dates = returns.index[window:]
gmvp_weights, utility_weights = [], []
mu_list, sigma_list = [], []

risk_aversion = 3  # Utility function parameter

# 🔹 Step (c) + (d) – Estimation and Optimization
for t in range(len(rolling_dates)):
    data = returns.iloc[t:t+window]
    mu_t = data.mean().values
    sigma_t = data.cov().values
    mu_list.append(mu_t)
    sigma_list.append(sigma_t)

    x = cp.Variable(n_assets)

    # GMVP
    gmvp_prob = cp.Problem(cp.Minimize(cp.quad_form(x, sigma_t)),
                           [cp.sum(x) == 1, x >= 0])
    gmvp_prob.solve()
    gmvp_weights.append(x.value)

    # Utility-maximizing portfolio
    util_prob = cp.Problem(cp.Maximize(mu_t @ x - 0.5 * risk_aversion * cp.quad_form(x, sigma_t)),
                           [cp.sum(x) == 1, x >= 0])
    util_prob.solve()
    utility_weights.append(x.value)

# 🔹 Step (e) – Convert results to DataFrame
gmvp_df = pd.DataFrame(gmvp_weights, index=rolling_dates, columns=assets)
util_df = pd.DataFrame(utility_weights, index=rolling_dates, columns=assets)

# 🔹 Step (e) – Plot portfolio weight evolution
plt.figure(figsize=(14, 6))
gmvp_df.plot(title='GMVP Weights Over Time', figsize=(14, 6), linewidth=2)
plt.ylabel("Weight")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
util_df.plot(title='Utility Portfolio Weights Over Time', figsize=(14, 6), linewidth=2)
plt.ylabel("Weight")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()
