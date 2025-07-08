# 📦 MODULE 2 & 3 – Market Regime Clustering + ML Forecasting of Expected Returns

"""
Author: Fabiano — Master in Quantitative Finance (CAU Kiel)
GitHub Repo: Dynamic Portfolio Optimization

This single Python module now bundles **Module 2 (Regime Clustering)** and **Module 3 (ML-based Return Forecasting)**.
It can be imported as a library (`import modules.portfolio_ml as pm`) or run as a standalone notebook.

Dependencies (Colab / pip):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn cvxpy
```

-------------------------------------------------------------------------------
MODULE CONTENTS
-------------------------------------------------------------------------------
1. `extract_features()`  – builds rolling-window features (vol, mean, corr)
2. `cluster_regimes()`   – KMeans clustering + PCA visualisation
3. `build_lagged_df()`   – creates lagged-return predictors (supervised ML)
4. `train_multitask_lasso()` – fits MultiTaskLassoCV to predict μ̂_{t+1}
5. `rolling_forecast_weights()` – optimises portfolio using predicted μ̂
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskLassoCV
import cvxpy as cp

# ---------------------------------------------------------------------------------
# SECTION 0 – Assume 'returns' exists (generated in Module 1) -----------------------
# ---------------------------------------------------------------------------------
# If running standalone, uncomment to simulate synthetic monthly MSCI-like returns.
# from utils.synthetic_data import get_synthetic_returns
# returns = get_synthetic_returns()

# ---------------------------------------------------------------------------------
# SECTION 1 – FEATURE EXTRACTION FOR REGIME CLUSTERING -----------------------------
# ---------------------------------------------------------------------------------
WINDOW = 60  # 60-month rolling window

def extract_features(ret: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Return DataFrame of engineered features for each rolling window."""
    feats, idx = [], ret.index[window:]
    for t in range(len(idx)):
        w = ret.iloc[t:t + window]
        sigma = w.cov().values
        vol   = np.sqrt(np.diag(sigma))               #  🔸 volatilities
        mean  = w.mean().values                       #  🔸 expected returns
        corr  = w.corr().values[np.triu_indices_from(sigma, k=1)]  # 🔸 pairwise corr
        feats.append(np.concatenate([vol, mean, corr]))
    cols = [f"f{i}" for i in range(len(feats[0]))]
    return pd.DataFrame(feats, index=idx, columns=cols)

# ---------------------------------------------------------------------------------
# SECTION 2 – REGIME CLUSTERING + VISUALISATION ------------------------------------
# ---------------------------------------------------------------------------------

def cluster_regimes(features: pd.DataFrame, k: int = 3):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    # Fit KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X)

    # PCA for 2-D visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot clusters in PCA space
    plt.figure(figsize=(9, 6))
    for i in range(k):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f"Regime {i}", alpha=0.7)
    plt.title("Market Regimes via KMeans (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot label time-series
    regime_series = pd.Series(labels, index=features.index, name="Regime")
    plt.figure(figsize=(14, 3))
    regime_series.plot(drawstyle="steps-post", color="darkblue")
    plt.title("Detected Market Regimes over Time")
    plt.xlabel("Date")
    plt.yticks(range(k))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return regime_series

# ---------------------------------------------------------------------------------
# SECTION 3 – SUPERVISED ML FORECASTING OF μ_{t+1} (Module 3) ----------------------
# ---------------------------------------------------------------------------------
LAGS = 12  # use past 12 months as predictors

def build_lagged_df(ret: pd.DataFrame, lags: int = LAGS) -> pd.DataFrame:
    """Return wide DataFrame where columns are asset_lagX."""
    lagged = {}
    for lag in range(1, lags + 1):
        lagged_df = ret.shift(lag).add_suffix(f"_lag{lag}")
        lagged[lag] = lagged_df
    full = pd.concat(lagged.values(), axis=1)
    full.dropna(inplace=True)
    return full


def train_multitask_lasso(ret: pd.DataFrame, lags: int = LAGS):
    """Train MultiTaskLassoCV to predict next-month returns for all assets."""
    X = build_lagged_df(ret, lags)
    y = ret.loc[X.index]  # align targets

    model = MultiTaskLassoCV(cv=5, random_state=42)
    model.fit(X.values, y.values)
    print("Best alpha:", model.alpha_)
    return model, X.index  # return fitted model + training index

# ---------------------------------------------------------------------------------
# SECTION 4 – ROLLING FORECAST + MEAN-VARIANCE OPTIMISATION ------------------------
# ---------------------------------------------------------------------------------

RISK_AVERSION = 3

def optimise_portfolio(mu_vec, sigma_mat):
    """Utility-maximising portfolio with no short-selling."""
    n = len(mu_vec)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(mu_vec @ x - 0.5 * RISK_AVERSION * cp.quad_form(x, sigma_mat)),
                      [cp.sum(x) == 1, x >= 0])
    prob.solve()
    return x.value


def rolling_forecast_weights(ret: pd.DataFrame, model, train_idx, window: int = WINDOW):
    weights, dates, valid_dates = [], ret.index[window:], []
    for t, date in enumerate(dates):
        sigma = ret.iloc[t:t + window].cov().values
        if date not in train_idx:
            continue
        X_pred = build_lagged_df(ret, LAGS).loc[date].values.reshape(1, -1)
        mu_hat = model.predict(X_pred).flatten()
        w_opt = optimise_portfolio(mu_hat, sigma)
        weights.append(w_opt)
        valid_dates.append(date)

    weights_df = pd.DataFrame(weights, index=valid_dates, columns=ret.columns)
    return weights_df

# ---------------------------------------------------------------------------------
# MAIN EXECUTION WRAPPER (example) -------------------------------------------------
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure 'returns' exists in the namespace
    try:
        returns
    except NameError:
        raise RuntimeError("DataFrame 'returns' not found. Import or simulate returns before running.")

    # MODULE 2 ------------------------------------------------
    feats = extract_features(returns)
    regimes = cluster_regimes(feats)

    # MODULE 3 ------------------------------------------------
    lasso_model, idx_train = train_multitask_lasso(returns)
    weights_ml = rolling_forecast_weights(returns, lasso_model, idx_train)

    # Visualise ML-driven weights
    weights_ml.plot(figsize=(14, 6), title="ML-Forecasted Portfolio Weights (Utility-Max)")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Export CSVs for GitHub
    regimes.to_csv("regime_labels.csv")
    weights_ml.to_csv("utility_weights_ml.csv")
