# Libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

params = {
'start': '2010-01-04',
'end': '2025-03-29', # not included; end -> '2025-03-28'
'tickers':['MSFT','AAPL','SPY'],
'auto_adjust': False,
'multi_level_index': False,
'progress': False
}

raw_data = yf.download(**params)

# NAV
date = '2025-03-28'
adj_close_dict = raw_data.loc['2025-03-28','Adj Close'].to_dict()
nav = 1e-5 + (500 * adj_close_dict['MSFT']) + (600 * adj_close_dict['AAPL']) + (-100 * adj_close_dict['SPY'])
print(f'Date: {date}\nNAV = ${nav:.2f}')

## Gross Leverage
numerator = (500 * adj_close_dict['MSFT']) + (600 * adj_close_dict['AAPL']) + abs(-100 * adj_close_dict['SPY'])
gross_leverage = numerator / nav
print(f'Date: {date}\nGross Leverage = {gross_leverage*100:.3f}%')

## Net Leverage
numerator = (500 * adj_close_dict['MSFT']) + (600 * adj_close_dict['AAPL']) - abs(-100 * adj_close_dict['SPY'])
net_leverage = numerator / nav
print(f'Net Leverage = {net_leverage*100:.3f}%')

# Data
df = raw_data.loc[:,'Adj Close']
df = df.reset_index(drop=False)
df.columns = ['Date', 'AAPL', 'MSFT', 'SPY']

df['NAV'] = np.nan
df['GROSS_LEVERAGE'] = np.nan
df['NET_LEVERAGE'] = np.nan

# Calculate - NAV, NET & GROSS LEVERAGE.
portfolio = {'AAPL':600, 'MSFT':500, 'SPY':-100}
r = 0.02
cash = 1e5
for i in range(df.shape[0]):
    nav = cash
    gross_leverage = 0
    net_leverage = 0
    for ticker in ['AAPL','MSFT','SPY']:
        nav += portfolio[ticker] * df.loc[i,ticker]
        gross_leverage += abs(portfolio[ticker] * df.loc[i,ticker])
        if ticker == 'SPY':
            net_leverage -= abs(portfolio[ticker] * df.loc[i,ticker])
        else:
            net_leverage += portfolio[ticker] * df.loc[i,ticker]
    df.loc[i,'NAV'] = nav
    growth_factor = (1 + r/365)
    cash = cash * growth_factor # Grow cash

    df.loc[i, 'GROSS_LEVERAGE'] = gross_leverage/nav
    df.loc[i, 'NET_LEVERAGE'] = net_leverage/nav

fig, ax = plt.subplots(1,2,figsize=(16,5))

ax[0].plot(df['Date'], df['NAV'], label = 'Portfolio NAV', color='blue')
ax[0].legend()

ax[1].plot(df['Date'], df['NET_LEVERAGE'], label = 'NET_LEVERAGE', color='blue')
ax[1].plot(df['Date'], df['GROSS_LEVERAGE'], label = 'GROSS_LEVERAGE', color='black')
ax[1].legend()
ax[1].set_title('Portfolio Leverage')
ax[1].set_ylabel('Leverage (%)')

plt.savefig('portfolio_nav_and_leverage.png')
plt.show()

# 90-day rolling volatility
df['LOG_PORTFOLIO_RETURNS'] =  np.log(df['NAV']/df['NAV'].shift(1))
df['90_DAY_ROLLING_VOLATILITY'] = df['LOG_PORTFOLIO_RETURNS'].rolling(window=90).std(ddof=1)
df['90_DAY_ROLLING_VOLATILITY_ANNUALIZED'] = df['90_DAY_ROLLING_VOLATILITY'] * np.sqrt(250)

plt.plot(df['Date'], df['90_DAY_ROLLING_VOLATILITY'], color='purple', label='ROLLING_VOLATILITY')
plt.plot(df['Date'], df['90_DAY_ROLLING_VOLATILITY_ANNUALIZED'], color='black', label='ROLLING_VOLATILITY_ANNUALIZED')
plt.legend(loc='upper left')
plt.title('90-Day Rolling Volatility')
plt.ylabel('Volatility')
plt.savefig('90_day_rolling_vol')
plt.show()

# df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
# df.to_excel('math5010-hw4-pb4.xlsx', index=False)