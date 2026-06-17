# Stock_Prediction
  Machine Learning in Trading Stocks - XGBoost classifier for NVDA next-day direction prediction. 


  ## Overview

  This notebook applies a **gradient-boosted classifier (XGBoost)** to predict the
  **next-day price direction** of NVIDIA Corp. (NVDA) using only historical price and
  volume data.

  Rather than forecasting price levels — a notoriously noisy regression task — the problem
  is framed as **binary classification**:

  $$y_t = \begin{cases} 1 & \text{if } P_{t+1} > P_t \quad \text{ (UP) } \\ 0 & \text{otherwise} \quad \text{ (DOWN) }
  \end{cases}$$

  A **confidence filter** (threshold > 55%) is applied at inference time to suppress
  low-conviction signals, simulating a no-trade zone common in systematic strategies.

  ---

  ## Dataset

  | Property      | Value                                    |
  |---------------|------------------------------------------|
  | Source        | `yfinance` — Yahoo Finance               |
  | Asset         | NVDA (NVIDIA Corporation)                |
  | Period        | 1999-11-05 → 2026-06-15                  |
  | Granularity   | Daily OHLCV                              |
  | Training set  | All data before the last 500 trading days |
  | Test set (OOS)| Last 500 trading days (~2 calendar years)|

  ---

  ## Feature Engineering

  Three families of features are constructed to capture different market dynamics.

  ### 1. Multi-Horizon Price Ratios
  Capture momentum and mean-reversion across multiple time scales:

  $$\text{CloseRatio}_h = \frac{P_t}{\overline{P}_{t,h}}, \quad h \in \{1, 2, 5, 21, 55, 100, 200\}$$

  ### 2. Rolling Trend Labels
  Count the number of UP days over the past $h$ sessions (lagged by 1 day to prevent leakage):

  $$\text{Trend}_h = \sum_{i=1}^{h} y_{t-i}$$

  ### 3. Technical Indicators via `pandas-ta`

  | Indicator        | Parameters                         |
  |------------------|------------------------------------|
  | RSI              | Length = 14                        |
  | Bollinger Bands  | Length = 20, std = 2               |
  | MACD             | Fast = 12, Slow = 26, Signal = 9   |

  > **Data leakage prevention:** returns are computed as
  > $R_t = (P_t - P_{t-1}) / P_{t-1}$ and the target label uses $P_{t+1}$,

  ---

  ## Model

  **XGBoost Classifier** with explicit regularization to counter overfitting on
  financial time series — the primary failure mode when signal-to-noise is low:

  ```python
  XGBClassifier(
      n_estimators  = 50,
      max_depth     = 3,     # shallow trees: weak learners by design
      learning_rate = 0.1,
      gamma         = 1.0,   # minimum loss reduction to make a split
      subsample     = 0.7,   # row subsampling per tree
      reg_alpha     = 0.1,   # L1 (Lasso) regularization
      reg_lambda    = 1.0,   # L2 (Ridge) regularization
      eval_metric   = 'logloss'
  )

  Shallow trees (max_depth=3) and stochastic subsampling deliberately reduce model
  capacity — in equity markets, overfitting is almost always the primary risk.

  ---
  Results

  Overall Performance

  ┌───────────────┬──────────┐
  │     Split     │ Accuracy │
  ├───────────────┼──────────┤
  │ In-Sample     │ 63%      │
  ├───────────────┼──────────┤
  │ Out-of-Sample │ 54%      │
  └───────────────┴──────────┘

                precision    recall  f1-score   support
          DOWN       0.51      0.27      0.35       235
            UP       0.54      0.77      0.64       265
      accuracy                           0.54       500

  Confidence-Filtered Predictions (last 20 trading days, threshold = 55%)

  ┌─────────────────────┬─────────┐
  │       Metric        │  Value  │
  ├─────────────────────┼─────────┤
  │ Signals triggered   │ 6 / 20  │
  ├─────────────────────┼─────────┤
  │ No-trade days       │ 14 / 20 │
  ├─────────────────────┼─────────┤
  │ Accuracy on signals │ 50%     │
  └─────────────────────┴─────────┘

  Interpretation

  54% OOS accuracy sits above the 50% random baseline — a non-trivial result in
  equity direction forecasting, where sustained edge of even 1–2% is considered
  meaningful in quantitative research. The IS/OOS gap (63% → 54%) is expected and
  reflects healthy generalization; a negligible gap would indicate underfitting.

  The confidence filter aggressively reduces trade frequency (6/20 signals) but
  does not yet improve precision — a known limitation addressed in the next episode.

  ---
  Limitations

  - No transaction costs, slippage, or market impact are modeled. Accuracy above 50%
  does not imply profitability.
  - A single holdout split is used. Proper walk-forward validation (expanding window)
  is on the roadmap.
  - The model is purely technical. Macro, fundamental, and sentiment signals are
  deliberately excluded in Episode 1 to establish a clean baseline.
  - The 55% confidence threshold is heuristic. Episode 2 explores probability
  calibration and dynamic filtering.

  ---
  Series Roadmap — From Signals to Alpha

  ┌─────────┬──────────────────────────────────────────────┬───────────────┐
  │ Episode │                    Topic                     │    Status     │
  ├─────────┼──────────────────────────────────────────────┼───────────────┤
  │ 1       │ NVDA Direction Classifier — XGBoost Baseline │ This notebook │
  ├─────────┼──────────────────────────────────────────────┼───────────────┤
  │ 2       │ Walk-Forward Validation + Feature Importance │ Coming soon   │
  ├─────────┼──────────────────────────────────────────────┼───────────────┤
  │ 3       │ Multi-Asset Expansion (AAPL, TSLA, SPY)      │ Planned       │
  ├─────────┼──────────────────────────────────────────────┼───────────────┤
  │ 4       │ Sequence Models — LSTM / Transformer         │ Planned       │
  ├─────────┼──────────────────────────────────────────────┼───────────────┤
  │ 5       │ Backtesting Framework + Strategy P&L         │ Planned       │
  └─────────┴──────────────────────────────────────────────┴───────────────┘

  ---
  Tech Stack

  ---
  Tech Stack

  Python 3.10+
  yfinance        market data retrieval
  pandas / numpy  data manipulation
  pandas-ta       technical indicator library
  xgboost         gradient boosted trees
  scikit-learn    metrics and evaluation
  matplotlib      visualization

  ---
  Setup

  git clone https://github.com/<your-username>/<repo-name>.git
  cd <repo-name>
  pip install -r requirements.txt
  jupyter notebook predict_NVDA.ipynb

  ---
  Author

  Simone Buzzo
  LinkedIn · GitHub

  ---
  Python 3.10+
  yfinance        market data retrieval
  pandas / numpy  data manipulation
  pandas-ta       technical indicator library
  xgboost         gradient boosted trees
  scikit-learn    metrics and evaluation
  matplotlib      visualization

  ---
  Setup

  git clone https://github.com/<your-username>/<repo-name>.git
  cd <repo-name>
  pip install -r requirements.txt
  jupyter notebook predict_NVDA.ipynb

  ---
  Author

  Simone Buzzo — Data Scientist
  LinkedIn · GitHub

  ---
