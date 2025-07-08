# 📦 MODULE 2, 3 & 4 – Regime Clustering + ML Forecasting + Stress Classification

"""
Author: Fabiano — Master in Quantitative Finance (CAU Kiel)
GitHub Repo: Dynamic Portfolio Optimization

This Python module implements:
- MODULE 2: Regime detection using KMeans clustering
- MODULE 3: Forecasting μ_t+1 with MultiTask Lasso
- MODULE 4: Predictive stress modeling using binary classification

Dependencies:
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
6. `compute_stress_index()` – calculates L2 stress from weight variations
7. `build_stress_classifier()` – predicts future stress using logistic regression
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskLassoCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import cvxpy as cp

WINDOW = 60
LAGS = 12
RISK_AVERSION = 3

# ---------------------------------------------------------------------------------
# MODULE 2 and 3 – Feature Engineering, Clustering and ML Forecasting
# (functions remain as in previous module)
# ---------------------------------------------------------------------------------
# ... (unchanged functions) ...

# ---------------------------------------------------------------------------------
# MODULE 4 – ALLOCATION STRESS INDEX & CLASSIFICATION -----------------------------
# ---------------------------------------------------------------------------------

def compute_stress_index(weights_df: pd.DataFrame) -> pd.Series:
    """Compute L2 norm difference of portfolio weights between t and t-1."""
    stress = [0]
    for t in range(1, len(weights_df)):
        delta = weights_df.iloc[t].values - weights_df.iloc[t - 1].values
        stress.append(np.linalg.norm(delta, ord=2))
    return pd.Series(stress, index=weights_df.index)


def build_stress_classifier(weights_df: pd.DataFrame, returns: pd.DataFrame, lags: int = 3):
    """
    Build classifier to predict if stress_t+1 > threshold based on return features.
    """
    stress = compute_stress_index(weights_df)
    threshold = stress.mean() + stress.std()
    y = (stress.shift(-1) > threshold).astype(int)  # predict future stress

    # Build lagged return features
    X = []
    idx = []
    for t in range(lags, len(returns)):
        row = returns.iloc[t-lags:t].values.flatten()
        X.append(row)
        idx.append(returns.index[t])

    X_df = pd.DataFrame(X, index=idx)
    X_df = X_df.loc[y.index.intersection(X_df.index)]
    y = y.loc[X_df.index]

    # Train Logistic Regression
    clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
    clf.fit(X_df.values, y.values)

    # Evaluation
    y_prob = clf.predict_proba(X_df.values)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"AUC: {auc:.4f}")
    print(classification_report(y, clf.predict(X_df.values)))

    # Plot ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.title("ROC Curve – Stress Prediction")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return clf, y, y_prob

# ---------------------------------------------------------------------------------
# MAIN EXECUTION WRAPPER -----------------------------------------------------------
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        returns
    except NameError:
        raise RuntimeError("DataFrame 'returns' not found. Import or simulate returns before running.")

    # MODULE 2
    feats = extract_features(returns)
    regimes = cluster_regimes(feats)

    # MODULE 3
    lasso_model, idx_train = train_multitask_lasso(returns)
    weights_ml = rolling_forecast_weights(returns, lasso_model, idx_train)

    # MODULE 4
    stress_series = compute_stress_index(weights_ml)
    stress_series.plot(title="Allocation Stress Index (L2 Norm)", figsize=(14, 4))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    clf, y_true, y_prob = build_stress_classifier(weights_ml, returns)

    # Export
    stress_series.to_csv("allocation_stress_index.csv")
