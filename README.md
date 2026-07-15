# 📈 Quantora — Intelligent Stock Screener & Portfolio Builder

Quantora is a multi-module Python application that screens equities across
Indian and global markets, runs seven machine-learning tracks over
fundamental + technical features, and builds optimized portfolios from the
results — all inside an interactive Streamlit dashboard.

> ⚠️ **Educational project only.** Nothing here is investment advice. Market
> data is sourced from Yahoo Finance and may be delayed, incomplete, or
> wrong. Do your own research before making financial decisions.

<img width="1918" height="747" alt="image" src="https://github.com/user-attachments/assets/a74f87d7-bf77-49ec-a2d0-bcc7c883eb3f" />

![Quantora overview](#)

---

## Table of Contents

- [What it does](#what-it-does)
- [Screenshots](#screenshots)
- [ML Pipeline — 7 Tracks](#ml-pipeline--7-tracks)
- [Portfolio Optimization](#portfolio-optimization)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Colab Notebook](#colab-notebook)
- [Known Limitations](#known-limitations)
- [Disclaimer](#disclaimer)

---

## What it does

Quantora takes a universe of tickers (a preset index sample or a custom
list), pulls historical + fundamental data, engineers a feature table, and
runs it through a full ML + portfolio-construction pipeline in one click:

1. **Screen** the whole universe — cluster archetype, anomaly flag,
   outperformance probability, and expected downside risk, all in one
   sortable table.
2. **Deep-dive** into a single ticker — price chart with moving averages,
   nearest peers, and a SHAP explanation of why the model scored it the
   way it did.
3. **Build a portfolio** from the screen results using either Maximum
   Sharpe Ratio (MPT) or Risk Parity optimization.

---

## Screenshots

### Indian Market Screen

<img width="1918" height="747" alt="image" src="https://github.com/user-attachments/assets/0e6ec9c6-c7e5-415b-9a8d-b948a728d96c" />



### Single Ticker Deep-Dive

<img width="1907" height="836" alt="image" src="https://github.com/user-attachments/assets/375105e3-38c8-48f6-ba0d-16231e0ad659" />



### Portfolio Builder

<img width="1566" height="786" alt="image" src="https://github.com/user-attachments/assets/60e6d570-8c83-4176-ba14-03ecf596db99" />




---

## ML Pipeline — 7 Tracks

All implemented in `ml_engine.py`, run per-analysis over the selected
ticker universe:

| # | Track | Purpose |
|---|-------|---------|
| 1 | **PCA** | Reduces 15+ fundamental/technical features down to ~5 principal components |
| 2 | **KMeans Clustering** | Groups stocks into auto-labeled archetypes (High Growth, Stable Dividend, High Risk/High Beta, Undervalued Value) — plus per-stock percentile tags computed against the whole universe, so a stock isn't hidden by its cluster's average |
| 3 | **Classifier Bake-off** | Trains XGBoost, LightGBM, Random Forest, and Logistic Regression (each hyperparameter-tuned via `RandomizedSearchCV` on time-series folds) to estimate the probability a stock outperforms the chosen benchmark; the best-scoring model on out-of-sample AUC is used for production scoring |
| 4 | **Random Forest Regressor** | Predicts expected 30-day downside risk (max drawdown) |
| 5 | **KNN Peer Finder** | Surfaces the 5 nearest alternative stocks by fundamental similarity |
| 6 | **Isolation Forest** | Flags anomalous / decoupled stocks relative to the rest of the universe |
| 7 | **SHAP TreeExplainer** | Explains individual predictions from the winning classifier, feature by feature |

Feature engineering includes **cross-sectional percentile-rank features**
(e.g. "cheaper than 80% of peers today" rather than just the raw P/E),
which tends to be a more stable predictor of relative outperformance than
raw fundamentals across changing market regimes.

---

## Portfolio Optimization

Implemented in `portfolio.py`:

- **Maximum Sharpe Ratio (MPT)** — classic mean-variance optimization via
  SLSQP, with configurable per-asset weight bounds.
- **Risk Parity** — allocates so each asset contributes equal risk to the
  portfolio, rather than equal capital.

Both use **Ledoit-Wolf shrinkage** for the covariance matrix and shrunk
mean-return estimates (toward the cross-sectional grand mean), since raw
sample statistics from daily returns are noisy and tend to produce unstable
or overconcentrated portfolios. Optimization runs multiple random restarts
and keeps the best result, since SLSQP is a local optimizer.

---

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **ML:** scikit-learn, XGBoost, LightGBM, SHAP
- **Optimization:** SciPy (SLSQP)
- **Data:** yfinance (Yahoo Finance)
- **Core:** Python, pandas, numpy

---

## Project Structure

```
quantora/
├── app.py             # Streamlit dashboard (3 tabs: Screen / Deep-Dive / Portfolio)
├── data_pipeline.py    # Data ingestion, feature engineering, training panel builder
├── ml_engine.py         # 7 ML tracks (PCA, clustering, classifier, risk model, KNN, anomaly, SHAP)
├── portfolio.py         # Max-Sharpe and Risk-Parity portfolio optimization
└── requirements.txt
```

---

## Installation

```bash
git clone <your-repo-url>
cd quantora
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` should include at minimum:

```
streamlit
plotly
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
scipy
yfinance
```

---

## Running the App

```bash
streamlit run app.py
```

Then in the sidebar:

1. Pick a ticker universe (NIFTY 50 sample, S&P 500 sample, or a custom
   comma-separated list).
2. Set the benchmark index.
3. Click **🚀 Run Full Analysis**.
4. Explore the three tabs — Global Market Screen, Single Ticker Deep-Dive,
   and Portfolio Builder.

---

## Colab Notebook

A notebook version of the pipeline (useful for exploring the ML tracks
outside the Streamlit UI) is available here:

**[Quantora — Colab Notebook](https://colab.research.google.com/drive/19HqWBQGSLEEHDxEkn-OM6dlgNoutrsRx?usp=sharing)**

<!-- SCREENSHOT: Colab notebook output cells, if you want a visual preview here -->
<img width="1918" height="787" alt="image" src="https://github.com/user-attachments/assets/903b3e2b-3316-44d6-bc64-6f7a89c24481" />


---

## Known Limitations

- Data completeness from Yahoo Finance varies by exchange — some fields
  (e.g. current price) can be missing for certain tickers or outside
  market hours.
- Classifier AUC reflects historical relative-outperformance patterns;
  past performance of the model itself is not a guarantee of future
  accuracy.
- Small ticker universes limit how meaningful clustering, peer-finding,
  and anomaly detection can be — results improve with a broader universe.

---

## Disclaimer

Quantora is a personal/educational project for exploring applied ML and
portfolio theory. It is **not** financial advice, and no output from this
tool should be the sole basis for an investment decision.
