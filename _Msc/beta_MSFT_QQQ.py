# %pip install pandas_datareader --quiet
# Necessary libraries.
import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importing the data.
tickers_ = ['QQQ','MSFT']
start_date = '2024-03-31'
end_date = '2025-03-31'
stock_data = yf.download(tickers=tickers_, start=start_date, end=end_date, auto_adjust=False)

# Calculating the log returns.
def log_returns(ticker, data):
    prices = stock_data.loc[:,('Adj Close',ticker)]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns

# SOFR Rates
sofr_df = web.DataReader('SOFR', 'fred', start=start_date, end=end_date)
plt.plot(sofr_df, color = 'black')
plt.title('SOFR rates')
plt.ylabel('Rates')
plt.xlabel('Date');
plt.savefig('SOFR rates')

avg_sofr = sofr_df.mean().values[0]
print(f'The average SOFR rate from {start_date} to {end_date} = {avg_sofr}')

risk_free_daily = avg_sofr / 250  # SOFR rate annualized
returns_df = pd.DataFrame({'MSFT returns':log_returns('MSFT', stock_data), 
              'QQQ returns':log_returns('QQQ', stock_data)
}) 

# Regression for Beta.
QQQ_returns = returns_df['QQQ returns'].values.reshape(-1, 1)
MSFT_returns = returns_df['MSFT returns'].values.reshape(-1, 1)
excess_QQQ = QQQ_returns - risk_free_daily
excess_MSFT = MSFT_returns - risk_free_daily

model = LinearRegression().fit(excess_QQQ, excess_MSFT)
beta = model.coef_[0][0]
alpha = model.intercept_[0]
pred = model.predict(excess_QQQ)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(excess_MSFT, color='red', label='Excess MSFT Returns')
ax[0].plot(excess_QQQ, color='black', label='Excess QQQ Returns')
ax[0].legend(loc='lower left')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Excess Returns')
ax[0].set_title('QQQ & MSFT Excess Returns')

ax[1].scatter(excess_QQQ,excess_MSFT, color = 'black', alpha = 0.7)
ax[1].plot(excess_QQQ, pred, color='red', label='LS line')
ax[1].set_xlabel('QQQ Daily Return')
ax[1].set_ylabel('MSFT Daily Return')
ax[1].set_title('MSFT vs QQQ - Linear Regression Fit')
ax[1].legend();

plt.savefig('returns_and_beta')

print(f"Beta (slope): {beta:.8f}")
print(f"Alpha (intercept): {alpha:.8f}")
