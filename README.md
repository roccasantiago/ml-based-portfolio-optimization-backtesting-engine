# ML-Based Portfolio Optimization & Backtesting Engine

This project implements an automated portfolio management system that combines **Modern Portfolio Theory (MPT)** with **Machine Learning (XGBoost)** algorithms. The goal is to outperform traditional asset allocation strategies (such as historical mean-variance) through dynamic prediction of expected returns.

The system performs a complete cycle: data ingestion, statistical cleaning, feature engineering, model training, convex optimization, and validation through *Walk-Forward Backtesting* with real-world transaction costs.

## Index
- [Project Objective](#project-objective)
- [Theoretical Foundation](#theoretical-foundation)
- [Methodology and Pipeline](#methodology-and-pipeline)
- [Installation & Usage](#installation--usage)
- [Code Structure](#code-structure)

---

## Project Objective

The primary goal is to answer the question: **Can a non-linear regression model improve the estimation of the expected returns vector ($\mu$) in the Markowitz Efficient Frontier?**

The project compares two strategies:
1.  **Classic Strategy:** Uses average historical returns as an estimator for future returns.
2.  **ML Strategy (XGBoost):** Uses a *Gradient Boosting* model to predict the next period's return based on technical indicators and macroeconomic variables.

Both strategies are benchmarked against the **S&P 500 (SPY)** index.

---

## Theoretical Foundation

### 1. Mean-Variance Optimization (Markowitz)
The core of the asset allocator seeks to maximize the Sharpe Ratio. Mathematically, we solve the following convex optimization problem:

$$
\begin{aligned}
\max_{w} \quad & \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}} \\
\text{s.t.} \quad & \sum w_i = 1 \\
& w_i \geq 0 \quad (\text{Long-only})
\end{aligned}
$$

Where:
* **$w$**: Portfolio weights vector.
* **$\mu$**: Expected returns vector (predicted by the ML model).
* **$\Sigma$**: Asset covariance matrix.
* **$r_f$**: Risk-free rate.

---

### 2. Robust Covariance Estimation
To avoid conditioning issues in the covariance matrix $\Sigma$ common when the number of assets **$N$** is large relative to the number of observations **$T$** we use **Ledoit-Wolf shrinkage**:

$$\hat{\Sigma}_{LW} = (1 - \alpha)S + \alpha F$$

**Where:**
* **$S$**: The Sample Covariance Matrix (the raw empirical covariance).
* **$F$**: The "Shrinkage Target" (a structured matrix where all pairwise correlations are equal to the average sample correlation).
* **$\alpha$**: The shrinkage intensity (a parameter between 0 and 1 that balances $S$ and $F$ to minimize Mean Squared Error).



---

### 3. Machine Learning (XGBoost)
Unlike linear models, XGBoost captures non-linear relationships between technical indicators (RSI, Momentum, Volatility) and future returns. 

The problem is defined as a regression where the **target** is the log return shifted to **$t+1$**

* **$t$**: Current time step (using information available today).
* **$t+1$**: The next period (prediction horizon).
* **$N$**: Dimension of the feature space (number of tickers).
* **$T$**: Total look-back window or training samples.

By predicting $\hat{\mu}_{t+1}$ with XGBoost, we feed the optimizer with a forward-looking return estimate instead of relying solely on historical averages.

---

## Methodology and Pipeline

The workflow is divided into sequential modules:

### 1. Ingestion and Preprocessing (`ingestion.py`)
* **Sources:** Yahoo Finance data (`yfinance`).
* **Cleaning:** Outlier filtering using a $3\sigma$ threshold (Z-score) to prevent "black swan" events from distorting training.
* **Macro Variables:** Inclusion of the Fear Index (**VIX**) and Treasury Bonds (**TNX**) to provide global market context.

### 2. Feature Engineering
Technical indicators are calculated to feed the model:
* **Momentum:** 5-day and 21-day rolling returns.
* **Trend:** Distance to the Simple Moving Average (SMA 20).
* **Volatility:** ATR (Average True Range) and VIX volatility.
* **Oscillators:** RSI (Relative Strength Index).

### 3. Modeling (`model.py`)
* **XGBoost Regressor** is utilized.
* **Cross-Validation:** `GridSearchCV` is implemented with temporal validation (Time Series Split implicit in the design) to optimize hyperparameters (`max_depth`, `learning_rate`, `n_estimators`).
* **Target:** `log_ret.shift(-1)`.

### 4. Optimization (`optimization.py`)
* Uses the `PyPortfolioOpt` library.
* **L2 Regularization** ($\gamma=0.1$) is applied to the objective function to avoid "corner solutions" (where the optimizer assigns 100% to a single asset) and to diversify risk.

### 5. Walk-Forward Backtesting (`backtesting.py`)
Realistic simulation that progresses window by window (monthly):
1.  Trains the model with data up to $t$.
2.  Optimizes weights for $t+1$.
3.  Calculates real returns in period $t+1$.
4.  Deducts **broker commissions** (0.1% per rebalance).

---

## Installation & Usage

1. **Clone & Setup:**
   ```
   git clone https://github.com/RoccaSantiago/Automatic-Porfolio.git
   cd portfolio-optimization
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
    ```
## 2. Configuration (`config.py`)

Before running the engine, update the `config.py` file to define your execution parameters:

* **Ticker Selection:** Define the list of assets to be analyzed.
* **Timeframe:** Select the start and end dates for data ingestion.
* **Feature Engineering:** Choose the indicators for the ML model. Available options include:
    * `rsi`, `atr`, `mom_5`, `mom_21`, `sma_dist`, `vix`, `tnx`, `vix_vol`.
* **Trading Costs:** Input the broker commission fee.
* **Output Settings:** Choose whether to save the generated dashboard or the processed dataset.

## 3. Execution

Once configured, run the main script to start the engine:

```bash
python main.py

---
## Code Structure

```bash
├── config.py           # Global configuration
├── ingestion.py        # Download, cleaning, and calculations
├── model.py            # XGBoost model definition and training
├── optimization.py     # Portfolio optimization logic
├── backtesting.py      # Walk-Forward simulation engine
└── main.py             # Execution and results visualization
```

