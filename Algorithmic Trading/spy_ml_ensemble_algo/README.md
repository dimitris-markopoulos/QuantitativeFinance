# ML-Based Trading Strategy (4-Model Ensemble)

This algorithm implements a 4-model ensemble for generating SPY trading signals using supervised machine learning.

## Overview

The model combines the outputs of four classifiers:
- Multinomial Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Each model outputs class probabilities for a 3-class target: Buy (+1), Hold (0), or Sell (-1). Ensemble probabilities are weighted and aggregated to generate final trading signals.

## Pipeline

- Data preparation using `yfinance` and `ta`
- Feature engineering and selection
- Supervised model training and probability prediction
- Ensemble signal generation
- Backtesting vs SPY benchmark
- Visualizations: signal distribution, confusion matrix, and cumulative returns

## Usage

```python
from data_preparation_pipeline import FinancialMLPipeline
from ml_model import MLTradingModel

model = MLTradingModel()
model.run()
