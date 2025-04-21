# MSFT SVM Algorithmic Trading Strategy

This repository implements a machine learning pipeline to predict the next-day direction of Microsoft's (MSFT) adjusted closing return using a Support Vector Machine (SVM) classifier. The model incorporates macroeconomic and sector indicators and is validated using time-aware cross-validation to avoid look-ahead bias.

## Pipeline Overview

### 1. Data Collection + Feature Engineering
- Historical daily data is pulled from Yahoo Finance for:
  - MSFT, ^VIX (volatility index), SPY (S&P 500), and XLK (technology sector ETF)
- Timeframe: 2022-01-03 to 2025-01-03
- Adjusted close prices are used to compute log returns
- Calculates daily log returns for each ticker
- Computes exponentially weighted moving average (EWMA) for MSFT returns
- Builds a labeled dataset:
  - Target variable is +1 for next-day positive MSFT return, -1 otherwise

### 2. Data Preparation
- Chronological train-test split:
  - 90% of the data for training
  - 10% held out for final testing
- Features are standardized using `StandardScaler`

### 3. Baseline SVM Model
- Fits an SVM with RBF kernel using `C=1` and `gamma=1`
- Evaluates model using 5-fold `TimeSeriesSplit` cross-validation
- Reports mean validation accuracy

### 4. Hyperparameter Tuning
- Performs grid search over:
  - `C`: [0.1, 1.0, 10, 100]
  - `gamma`: [0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10]
- Uses `GridSearchCV` with `TimeSeriesSplit`
- Reports best-performing parameters and cross-validated accuracy

## Results
- Baseline model (C=1, gamma=1): ~50.7% cross-validated accuracy
- Tuned model (C=1, gamma=0.3): ~51.6% cross-validated accuracy
- The improvement, while small, demonstrates the importance of tuning even simple models
