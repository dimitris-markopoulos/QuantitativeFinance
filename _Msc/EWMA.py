# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
import datetime

ticker = 'AAPL'
df = yf.download(tickers = ticker, start = '2025-01-01', end = '2025-02-01', auto_adjust = False)
df['Returns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))

#Method 1 - Package

r_ewm  = df['Returns'].ewm(alpha=0.06, min_periods=1, adjust=True)
r_std = r_ewm.std(bias=True)
vol_package = r_std.tail(1).values[0]
print(f'\n\nVolatility Forecast w/ Package = {vol_package}')

#Method 2 - Manual

def calculate_ewma_volatility(returns, lambda_param=0.94):
    n = len(returns)
    weights = np.array([(1-lambda_param)/(1-lambda_param**n) * lambda_param**(i) for i in range(n - 1, -1, -1)])
    r_hat = np.sum(weights * returns)
    squared_deviations = (returns - r_hat)**2
    weighted_sq_dev = weights * squared_deviations
    variance = np.sum(weighted_sq_dev)
    volatility = np.sqrt(variance)
    return volatility

returns = df['Returns'][1:]

vol_manual = calculate_ewma_volatility(returns)

print(f'Volatility Forecast Manually = {vol_manual}')
print(f'Difference = {(vol_manual - vol_package):.10f}')

assert vol_manual - vol_package < 10**-6, 'Error'
