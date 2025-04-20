# Notebook 1 - Data Preparation
from data_preparation_pipeline import FinancialMLPipeline

# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Machine Learning Packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class MLTradingModel:

    def __init__(self, ensemble_weights=None,buy_threshold=0.005,sell_threshold=0.001,test_start='2025-01-20',xgb_max_depth=3,xgb_learn_rate=0.04,light_max_depth=3,light_learn_rate=0.03):
        if ensemble_weights is None:
            ensemble_weights = {'lr': 0.25, 'rf': 0.25, 'xgb': 0.25, 'lgb': 0.25}
        self.weights = ensemble_weights
        self.models = {}
        self.selected_features = []
        self.buy_threshold=buy_threshold
        self.sell_threshold=sell_threshold
        self.test_start=test_start
        self.pipeline = FinancialMLPipeline()
        self.df_full=self.pipeline.run(buy_threshold=buy_threshold,sell_threshold=sell_threshold,test_start=test_start)
        self.xgb_max_depth=xgb_max_depth
        self.xgb_learn_rate=xgb_learn_rate
        self.light_max_depth = light_max_depth
        self.light_learn_rate=light_learn_rate

    def get_X_y(self, df, set_name):
        subset = df[df['set'] == set_name]
        X = subset.drop(columns=['Date', 'Target', 'set'])
        y = subset['Target']
        return X, y
   
    def run(self):
        X_tr, y_tr = self.pipeline.get_X_y(self.df_full,set_name='train')
        X_ts, y_ts = self.pipeline.get_X_y(self.df_full,set_name='test')

        # Scale for required models
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_ts_scaled = scaler.transform(X_ts)

        sfs = SFS(LogisticRegression(max_iter=1000),
          k_features=10,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=3,
          n_jobs=-1)

        sfs.fit(X_tr_scaled, y_tr)
        selected_features = [X_tr.columns[int(i)] for i in sfs.k_feature_names_]

        self.selected_features=selected_features

        X_tr_sub = X_tr[selected_features]
        X_ts_sub = X_ts[selected_features]

        X_tr_sub_scaled = scaler.fit_transform(X_tr_sub)
        X_ts_sub_scaled = scaler.transform(X_ts_sub)

        # Multinomial Regression
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(X_tr_sub_scaled, y_tr)
        lr_probs = lr.predict_proba(X_ts_sub_scaled)

        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        rf.fit(X_tr_sub, y_tr)
        rf_probs = rf.predict_proba(X_ts_sub)

        # 3. XGBoost
        label_map = {-1: 0, 0: 1, 1: 2} # Convert labels from [-1, 0, 1] → [0, 1, 2]
        y_tr_xgb = y_tr.map(label_map)
        y_val_xgb = y_ts.map(label_map)

        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=self.xgb_max_depth, learning_rate=self.xgb_learn_rate,
                                    objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')

        xgb_model.fit(X_tr_sub, y_tr_xgb)
        xgb_probs = xgb_model.predict_proba(X_ts_sub)

        # 4. LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=self.light_learn_rate, max_depth=self.light_max_depth)
        lgb_model.fit(X_tr_sub, y_tr)
        lgb_probs = lgb_model.predict_proba(X_ts_sub)

        w_lr  = self.weights['lr']
        w_rf  = self.weights['rf']
        w_xgb = self.weights['xgb']
        w_lgb = self.weights['lgb']

        assert abs(w_lr + w_rf + w_xgb + w_lgb - 1.0) < 1e-6, 'does not sum to 1'

        ensemble_probs = (w_lr  * lr_probs + w_rf  * rf_probs + w_xgb * xgb_probs + w_lgb * lgb_probs)

        class_labels = np.array([-1, 0, 1])
        signals_val = class_labels[np.argmax(ensemble_probs, axis=1)]

        unique, counts = np.unique(signals_val, return_counts=True)
        distribution = dict(zip(unique, counts))

        # Plot
        labels = sorted(distribution.keys())
        counts = [distribution[k] for k in labels]

        color_map = {-1: 'red', 0: 'grey', 1: 'green'}
        colors = [color_map[k] for k in labels]

        plt.bar(labels, counts, color=colors)
        total = sum(counts)
        for x, count in zip(labels, counts):
            percent = 100 * count / total
            plt.text(x, count + 0.2, f'{count} ({percent:.1f}%)', ha='center', va='bottom')
        plt.title('Signal Distribution')
        plt.xlabel('Signal')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('ensemble_signal_distribution.png')
        plt.show()

        # Generate confusion matrix
        cm = confusion_matrix(y_ts, signals_val, labels=[-1, 0, 1])

        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 0, 1])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix on Validation Set')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('confusion_matrix')
        plt.show()

        returns_val = self.df_full.loc[X_ts.index, 'Log_Return'].values
        dates_val = X_ts.index
        dates = self.df_full.loc[dates_val,'Date']

        strategy_returns = signals_val * returns_val # Strategy returns = signal × return
        cumulative_strategy = (strategy_returns + 1).cumprod()
        cumulative_market = (returns_val + 1).cumprod()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, cumulative_strategy, label='Strategy', color='blue', linewidth=2)
        plt.plot(dates, cumulative_market, label='SPY (Buy & Hold)', color='orange', linestyle='--', linewidth=2)
        plt.title('Cumulative PnL on Test Set')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig('cumulative_pnl')
        plt.show()

        return {
                'signals': signals_val,
                'strategy_returns': strategy_returns,
                'cumulative_strategy': cumulative_strategy,
                'cumulative_market': cumulative_market,
                'dates': dates
        }
