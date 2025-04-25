import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
import ta

class FinancialMLPipeline:

    def __init__(self, ticker='SPY'):
        self.ticker = ticker

    def get_price_data(self, ticker=None, start_date=None, end_date=None):
        if ticker is None:
            ticker = self.ticker

        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False, multi_level_index=False)
        df = df.reset_index()
        df['Date'] = df['Date'].astype('datetime64[ns]')

        # Include VIX
        vix = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=False, progress=False, multi_level_index=False)
        vix = vix.reset_index()[['Date', 'Adj Close']].rename(columns={'Adj Close': 'VIX'})
        vix['Date'] = pd.to_datetime(vix['Date'])

        df_merged = df.merge(vix, on='Date', how='left')

        # Include 10yr Treasury Yield
        tenyr = yf.download('^TNX', start=start_date, end=end_date, auto_adjust=False, progress=False, multi_level_index=False)
        tenyr = tenyr.reset_index()[['Date', 'Adj Close']].rename(columns={'Adj Close': 'TNX'})
        df = df_merged.merge(tenyr, on='Date', how='left')

        return df

    def features_engineer(self, df, rsi_window=14, stoch_window=14, sma_windows=(20, 50), ema_windows=(20,), bb_window=20, std_window=20, atr_window=14):
        df = df.copy()

        # Shift VIX and TNX to avoid data leakage
        df['Lag1_VIX'] = df['VIX'].shift(1)
        df['Lag1_TNX'] = df['TNX'].shift(1)
        df = df.drop(columns=['VIX', 'TNX'])

        # Momentum Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close'], window=rsi_window).rsi()
        macd = ta.trend.MACD(df['Adj Close'])
        df['MACD'] = macd.macd()
        df['Stoch_Osc'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=stoch_window).stoch()

        # Trend Indicators
        for window in sma_windows:
            df[f'SMA{window}'] = ta.trend.SMAIndicator(df['Adj Close'], window=window).sma_indicator()
        for window in ema_windows:
            df[f'EMA{window}'] = ta.trend.EMAIndicator(df['Adj Close'], window=window).ema_indicator()

        # Volatility Indicators
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_window).average_true_range()
        bb = ta.volatility.BollingerBands(df['Adj Close'], window=bb_window)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df[f'Rolling_Std_{std_window}'] = df['Adj Close'].rolling(window=std_window).std()

        # Volume Indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Adj Close'], df['Volume']).on_balance_volume()
        df['Volume_Change'] = df['Volume'].pct_change()

        # Return Features
        df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

        # Extra Features
        df['Daily_Range'] = df['High'] - df['Low']
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Gap'] = df['Open'] - df['Close'].shift(1)

        return df

    def create_trading_target(self, df, buy_threshold=0.005, sell_threshold=0.005):
        df = df.copy()
        df['Target'] = df['Log_Return'].shift(-1).apply(
            lambda x: 1 if x > buy_threshold else (-1 if x < -sell_threshold else 0)
        )
        return df

    def tag_time_splits(self, df, test_start='2025-03-20'):
        df = df.copy()
        df['set'] = 'train'
        df.loc[df['Date'] >= test_start, 'set'] = 'test'
        return df


    def get_X_y(self, df, set_name):
        subset = df[df['set'] == set_name]
        X = subset.drop(columns=['Date', 'Target', 'set'])
        y = subset['Target']
        return X, y

    def run(self, start_date=None, end_date=None,
            rsi_window=14, stoch_window=14,
            sma_windows=(20, 50), ema_windows=(20,),
            bb_window=20, std_window=20, atr_window=14,
            buy_threshold=0.005, sell_threshold=0.005,
            val_start='2019-01-01', test_start='2025-03-20',
            plot_split=True):

        df = self.get_price_data(start_date=start_date, end_date=end_date)

        df = self.features_engineer(
            df,
            rsi_window=rsi_window,
            stoch_window=stoch_window,
            sma_windows=sma_windows,
            ema_windows=ema_windows,
            bb_window=bb_window,
            std_window=std_window,
            atr_window=atr_window
        )

        df = df.dropna().reset_index(drop=True)
        df = self.create_trading_target(df, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        df = self.tag_time_splits(df, test_start=test_start)

        os.makedirs('media', exist_ok=True)
        if plot_split:
            plt.figure(figsize=(10, 2))
            plt.plot(df['Date'], [1]*len(df), '|', color='gray', alpha=0.5)

            for label, color in zip(['train', 'test'], ['green', 'orange']):
                subset = df[df['set'] == label]
                plt.plot(subset['Date'], [1]*len(subset), '|', label=label, color=color, markersize=12)

            plt.legend(loc='upper left', ncol=3)
            plt.title('Train/Test Time Split')
            plt.yticks([])
            plt.xlabel('Date')
            plt.tight_layout()
            plt.savefig('media/tr_ts_distribution')
            plt.show()

        return df
