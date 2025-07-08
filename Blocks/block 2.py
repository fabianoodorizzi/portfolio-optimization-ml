import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

# --- 📦 STEP 1: Simulate synthetic MSCI-like data ---
# Simulate 5 international indices from 2000 to 2025 at monthly frequency
np.random.seed(42)
dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='M')
n_assets = 5
asset_names = ["MSCI_USA", "MSCI_Europe", "MSCI_China", "MSCI_Japan", "MSCI_Brazil"]

# Generate synthetic price paths (random walk with small drift)
prices = pd.DataFrame(
    np.cumprod(1 + 0.005 * np.random.randn(len(dates), n_assets), axis=0) * 100,
    index=dates,
    columns=asset_names
)

# --- 🧮 STEP 2: Compute log-returns ---
returns = np.log(prices / prices.shift(1)).dropna()

# --- 🔁 STEP 3: Define rolling window parameters ---
window_size = 60  # 60 months = 5 years
rolling_dates = returns.index[window_size:]
gmvp_weights = []  # store optimal weights per window

# --- 📊 STEP 4: Rolling GMVP Optimization ---
for t in range(len(rolling_dates)):
    # Extract window of returns
    window_data = returns.iloc[t:t + window_size]

    # Compute sample covariance matrix
    sigma_t = window_data.cov().values

    # Define variable for asset weights
    x = cp.Variable(n_assets)

    # Define GMVP optimization problem: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(x, sigma_t))
    constraints = [cp.sum(x) == 1, x >= 0]  # fully invested, no short-selling
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Store weights
    gmvp_weights.append(x.value)

# Convert to DataFrame: one row per time window
gmvp_df = pd.DataFrame(gmvp_weights, index=rolling_dates, columns=asset_names)

# --- 📉 STEP 5: Compute Allocation Stress Index (L2 norm of Δweights) ---
stress_index = [0]  # stress at t=0 is 0 by definition
for t in range(1, len(gmvp_df)):
    diff = gmvp_df.iloc[t].values - gmvp_df.iloc[t - 1].values
    stress = np.linalg.norm(diff, ord=2)  # Euclidean norm
    stress_index.append(stress)

# Create a time series of stress values
stress_series = pd.Series(stress_index, index=gmvp_df.index)

# --- 📈 STEP 6: Visualize the stress index ---
plt.figure(figsize=(14, 5))
plt.plot(stress_series, label='Allocation Stress Index (L2 Norm)', color='tomato')
plt.axhline(stress_series.mean(), color='gray', linestyle='--', label='Mean Stress')
plt.title("📉 Allocation Stress Index Over Time (GMVP Changes)")
plt.xlabel("Date")
plt.ylabel("L2 Norm of Δweights")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 📁 STEP 7: Exportable DataFrame for GitHub or future modules ---
stress_df = pd.DataFrame({
    "Stress_L2": stress_series,
    "Mean_Stress": stress_series.mean(),
    "High_Stress_Flag": stress_series > (stress_series.mean() + stress_series.std())
})

# Simply display the DataFrame in Jupyter or Colab
display(stress_df.head(10))

