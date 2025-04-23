# Basics
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import matplotlib.dates as mdates

# ML
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

class SVMTradingPipeline:
    def __init__(self, ticker, start_date, end_date, test_prop=0.10):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.test_prop = test_prop
        self.df = None
        self.raw_data = None
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.signals = None

    @property
    def download_data(self):
        ticker_plus_vix_spy_xlk = [self.ticker,'^VIX','SPY','XLK']
        params = {
        'tickers':ticker_plus_vix_spy_xlk,
        'start':self.start_date,
        'end':self.end_date,
        'auto_adjust':False,
        'progress':False
        }
        raw_data = yf.download(**params)
        raw_data = raw_data.ffill()
        self.ticker_plus_vix_spy_xlk = ticker_plus_vix_spy_xlk
        self.raw_data = raw_data
        return raw_data

    @property
    def engineer_features(self):
        raw_data = self.raw_data
        tickers = self.ticker_plus_vix_spy_xlk
        engineered_dict = {f'{ticker}_logreturns':[] for ticker in tickers}
        for tick in tickers:
            adj = raw_data.loc[:,('Adj Close',tick)]
            log_returns = np.log(adj) - np.log(adj.shift(1))
            log_returns_vals = list(log_returns.values)
            engineered_dict[str(f'{tick}_logreturns')].extend(log_returns)

            if tick in [self.ticker]:
                ewma = log_returns.ewm(alpha=0.1, adjust=False).mean()
                engineered_dict[f'{tick}_ewm'] = ewma

        df = pd.DataFrame(engineered_dict, index=raw_data.index).dropna() # Drop NA

        # Targets -> {1:'next day positive returns', -1:'next day negative returns'}
        shifted = df[f'{self.ticker}_logreturns'].shift(-1)
        df['Target'] = shifted.apply(lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) else np.nan))
        df = df.dropna()
        self.df = df
        return df

    @property
    def split_data(self): # (X_tr, y_tr), (X_ts, y_ts)
        split = {'prop_tr':1 - self.test_prop,'prop_ts':self.test_prop}
        assert abs(sum(split.values()) - 1.0) < 1e-8, "Split proportions must sum to 1.0"

        n,p = self.df.shape
        n_tr = int(np.ceil(n*split['prop_tr']))
        self.n_tr = n_tr

        X = self.df.drop(columns='Target')
        y = self.df['Target']
        self.y = y

        X_tr, y_tr = X.iloc[:n_tr].to_numpy(), y.iloc[:n_tr].to_numpy()
        X_ts, y_ts = X.iloc[n_tr:].to_numpy(), y.iloc[n_tr:].to_numpy()

        scaler = StandardScaler() # Scale features
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_ts_scaled = scaler.transform(X_ts)

        self.X_tr_scaled = X_tr_scaled
        self.X_ts_scaled = X_ts_scaled
        self.y_tr = y_tr
        self.y_ts = y_ts

        return {'Training':[X_tr,y_tr], 'Testing': [X_ts,y_ts]}

    def cross_validate(self, svm_param_grid=None, n_splits=5, runtime='fast'):

        X_tr_scaled = self.X_tr_scaled
        y_tr = self.y_tr

        # Define default hyperparameter grid if not provided
        if svm_param_grid is None:
            if runtime == 'fast':
                svm_param_grid = {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                }
            elif runtime == 'slow':
                svm_param_grid = {
                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'C': [0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': ['scale', 'auto', 1e-3, 1e-2, 0.1, 1, 10],
                    'degree': [2, 3, 4],
                    'coef0': [0.0, 0.1, 0.5, 1.0],
                    'shrinking': [True, False],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'class_weight': [None, 'balanced'],
                }
            else:
                raise ValueError("runtime must be either 'fast' or 'slow'")

        svm_model = SVC()
        cv = TimeSeriesSplit(n_splits=n_splits)
        grid = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=cv, n_jobs=-1)
        grid.fit(X=X_tr_scaled, y=y_tr)

        best_params = grid.best_estimator_.get_params()
        best_params_subset = {key: best_params[key] for key in svm_param_grid.keys()}
        best_score = grid.best_score_

        self.best_params = best_params
        return best_params

    def fit_predict(self, param_grid = None, plot_distribution=False, save_path = None):
        if param_grid == None:
            param_grid = self.best_params

        tuned_model = svm_model = SVC(**param_grid).fit(self.X_tr_scaled,self.y_tr)
        y_pred = list(tuned_model.predict(self.X_ts_scaled))

        test_range = list(self.y.index)[self.n_tr].strftime('%Y-%m-%d'), list(self.y.index)[-1].strftime('%Y-%m-%d')
        self.test_range = test_range
        signals = pd.Series(y_pred, index=self.df.loc[test_range[0]:test_range[1],:].index)
        self.signals = signals
        print(f"Evaluate the model on test set range: {test_range[0]} to {test_range[1]}.")

        stock_prices_plus_one = self.raw_data['Adj Close'][self.ticker].iloc[self.n_tr:]
        previous_close = stock_prices_plus_one.shift(1).dropna()
        self.previous_close = previous_close


        if plot_distribution:
            labels_raw, counts = np.unique(y_pred, return_counts=True)
            label_map = {-1.0: 'Sell', 1.0: 'Buy'}
            labels = [label_map[val] for val in labels_raw]

            if save_path is None:
                raise ValueError("You must provide a save_path when plot_distribution is True.")
            
            plt.figure(figsize=(6, 4))
            plt.bar(labels, counts, color=['red', 'green'])
            plt.title('Model Signal Distribution')
            plt.ylabel('Count')
            plt.grid(axis='y', alpha = 0.3)
            plt.savefig('media/signal_distribution')
            plt.show()

            print(f"Plot saved to {save_path}")
        else:
            print("plot_distribution is False, skipping plot.")

    def simulate_trading(self, initial_capital=10000, benchmark=True):
        previous_close = self.previous_close
        # Model PnL Calculation
        initial = initial_capital
        PnL = []
        cash = initial
        shares = 0
        position = 0

        for date, signal in self.signals.items():
            date = date.strftime('%Y-%m-%d')
            price = previous_close[date]
            if position == 0 and signal == 1.0: # No shares & BUY -> BUY
                shares = cash / price
                cash = 0
                position = 1

            elif position == 1 and signal == -1.0: # Holding shares & SELL -> SELL
                cash = price*shares
                shares = 0
                position = 0

            else: # (No shares & SELL) | (Holding shares & BUY) -> No action
                pass

            current_value = cash + shares * price
            PnL.append((date, current_value))

        pnl_df = pd.DataFrame(PnL, columns=['Date','PnL'])
        df = pnl_df.copy()

        if benchmark == True:
            # Benchmark PnL
            prices = self.previous_close.iloc[:-1]
            shares = initial / self.previous_close[self.test_range[0]]
            benchmark_pnl = []

            for date, price in prices.items():
                date = date.strftime('%Y-%m-%d')
                pnl = shares * price
                benchmark_pnl.append((date,pnl))

            benchmark_df = pd.DataFrame(benchmark_pnl,columns=['Date','Benchmark PnL'])

            df = pd.merge(pnl_df, benchmark_df, on='Date', how='outer')
        
        return df
    
    def plot_pnl(self, joint_df, figsize = (12, 5), save_path='media/model_vs_benchmark_PnL.png'):
        joint_df['Date'] = pd.to_datetime(joint_df['Date'])
        plt.figure(figsize=figsize)
        plt.plot(joint_df['Date'], joint_df['PnL'], label='Model', color='black')
        if 'Benchmark PnL' in joint_df.columns:
            plt.plot(joint_df['Date'], joint_df['Benchmark PnL'], label='Benchmark', color='black', linestyle=':')
        final_date = joint_df['Date'].iloc[-1]
        final_date_label = f'Final Day: {final_date.strftime("%Y-%m-%d")}'
        plt.axvline(final_date, color='red', linestyle='--', linewidth=1.5, label=final_date_label)
        plt.ylabel('Portfolio Value ($)')
        plt.title(f'PnL Over Time - Model Strategy on {self.ticker}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.savefig(save_path)
        plt.show()
