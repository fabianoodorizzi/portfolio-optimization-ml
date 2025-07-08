# 📦 MODULE 2 – Clustering Regime Switching with KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- STEP 1: Feature Extraction from Rolling Windows ---
# Assumes 'returns' and 'gmvp_df' already exist from Module 1

window_size = 60
features = []
feature_dates = returns.index[window_size:]

for t in range(len(feature_dates)):
    window = returns.iloc[t:t+window_size]
    sigma = window.cov().values
    vol = np.diag(sigma) ** 0.5
    mean = window.mean().values
    corr = window.corr().values[np.triu_indices_from(sigma, k=1)]

    features.append(np.concatenate([vol, mean, corr]))

features_df = pd.DataFrame(features, index=feature_dates)

# --- STEP 2: Feature Standardization ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

# --- STEP 3: PCA for Visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- STEP 4: KMeans Clustering ---
k = 3  # Number of regimes (crisis, stable, volatile)
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
kmeans.fit(X_scaled)
labels = kmeans.labels_

# --- STEP 5: Visualization of Clusters in PCA space ---
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f"Regime {i}", alpha=0.7)
plt.title("Market Regimes via KMeans Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- STEP 6: Time Series Visualization of Regimes ---
regime_series = pd.Series(labels, index=feature_dates)
plt.figure(figsize=(14, 4))
regime_series.plot(drawstyle="steps-post", color="darkblue")
plt.title("Detected Market Regimes over Time")
plt.xlabel("Date")
plt.ylabel("Regime Label")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- STEP 7: Attach regimes to GMVP weights for analysis ---
gmvp_with_regime = gmvp_df.copy()
gmvp_with_regime["Regime"] = regime_series

# (Optional) Export to CSV for GitHub
# gmvp_with_regime.to_csv("gmvp_weights_with_regime.csv")
