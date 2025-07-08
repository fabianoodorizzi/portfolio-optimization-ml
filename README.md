# portfolio-optimization-ml
Dynamic Portfolio Optimization with Regime Clustering and Machine Learning Forecasting

# 🧠 Adaptive Portfolio Optimization via Market Regime Clustering (Python Project)
🎯 Project Aim

The goal of this project is to simulate international financial markets and develop a data-driven strategy that:

Builds optimal portfolios (GMVP and utility-based)

Detects structural shifts in market behavior (regimes)

Evaluates portfolio performance across regimes

By combining portfolio theory, unsupervised learning, and robust evaluation, the project demonstrates how a quantitative investor can adapt allocations to changing market dynamics.

🔍 Overview

The codebase is organized into 5 modular blocks, each corresponding to a distinct step in the process. Explanations are embedded directly in the Python files (blocks/blocco_1.py to blocco_5.py) for clarity.

The README provides a high-level map.

This repository implements an **adaptive portfolio optimization strategy** using a simulated financial time series with **multiple volatility regimes**. 
We use **unsupervised machine learning (KMeans clustering)** to identify hidden market states and then optimize portfolios for each regime based on covariance structure. 
The goal is to **adjust portfolio allocation dynamically** based on market conditions.

📌 This project bridges concepts from:
- **Portfolio Theory** (Markowitz, risk-return optimization)
- **Machine Learning** (clustering, regime-switching models)
- **Quantitative Finance** (covariance estimation, efficient frontier)

---

## 📁 Project Structure

- `blocco_1.txt` → Simulate multiregime time series (base + 2 volatility levels)
- `blocco_2.txt` → Compute rolling volatility and apply KMeans clustering
- `blocco_3.txt` → Estimate regime-specific covariance matrices
- `blocco_4.txt` → Optimize portfolio for each regime (GMVP)
- `blocco_5.txt` → Visualize time-varying portfolio weights and clustering
- `readme.md` → Project explanation and theoretical background (this file)

---

## 📊 Core Steps Implemented

### ✅ 1. **Synthetic Time Series Simulation**
- Simulates a multivariate return series with 3 regimes: low, medium, and high volatility.
- Uses `numpy.random.multivariate_normal` and block concatenation.

### ✅ 2. **Volatility Estimation and Clustering**
- Computes rolling volatility using standard deviation.
- Applies `KMeans` (from `sklearn.cluster`) to group into 3 regimes.

### ✅ 3. **Regime-Specific Covariance Estimation**
- For each detected cluster, computes the empirical covariance matrix.

### ✅ 4. **Portfolio Optimization (GMVP per regime)**
- Solves the Global Minimum Variance Portfolio problem:
  $$ \min_w w^T \Sigma w \quad \text{s.t.} \; \sum w_i = 1, \; w_i \ge 0 $$
- Uses `cvxpy` for convex optimization.

### ✅ 5. **Dynamic Allocation and Visualization**
- Allocates portfolio weights based on detected regime.
- Plots dynamic weights and clustering labels.

---

## 📈 Sample Outputs

- Clustering of time series into market regimes
- Covariance heatmaps per regime
- Portfolio weight trajectories based on detected regime

---

## ⚙️ Technologies Used

- Language: Python 3.11+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `cvxpy`
- IDE: Jupyter Notebook / Google Colab

---

## 🧠 Skills Demonstrated

| Area | Competence |
|------|------------|
| Portfolio Analysis | Efficient frontier, GMVP, risk-return tradeoff |
| Time Series Simulation | Multivariate normal, regime switching |
| Machine Learning | KMeans clustering, unsupervised learning |
| Optimization | Convex programming with constraints |
| Data Science | Rolling window stats, matrix manipulation, plotting |


---

## 📎 License

This project is released under the MIT License.

---
