import pandas as pd
import numpy as np

class StandardScoreConverter:
    def __init__(self, X_train):
        self.X_train = X_train
    
    def get_converted_data(self):
        X_train2 = (self.X_train).copy()
        feature_means = []
        feature_stds = []
        for col in list(X_train2.columns): #feature
            mean = (X_train2[col].values).mean()
            std = (X_train2[col].values).std(ddof=1)
            X_train2[col] = (X_train2[col] - mean) / std
            feature_means.append(mean)
            feature_stds.append(std)
            
        X_train_scaled = X_train2
        self.mean_ = np.array(feature_means)
        self.std_ = np.array(feature_stds)

        return X_train_scaled

    def convert(self, X_test):
        for i, col in enumerate(list(X_test.columns)):
            X_test[col] = (X_test[col] - self.mean_[i]) / self.std_[i]
        return X_test
        
class MinMaxConverter:
    def __init__(self, X_train):
        self.X_train = X_train

    def get_converted_data(self):
        X_train2 = (self.X_train).copy()
        feature_max = []
        feature_min = []
        for col in list(X_train2.columns): #feature
            max_ = (X_train2[col].values).max()
            min_ = (X_train2[col].values).min()
            X_train2[col] = (X_train2[col] - min_) / (X_train2[col] - max_)
            feature_max.append(max_)
            feature_min.append(min_)
            
        X_train_scaled = X_train2
        self.max_ = np.array(max_)
        self.min_ = np.array(min_)

        return X_train_scaled

    def convert(self, X_test):
        for i, col in enumerate(list(X_test.columns)):
            X_test[col] = (X_test[col] - self.min_[i]) / (X_test[col] - self.max_[i])
        return X_test
    
class BaseConverter:
    def __init__(self, X_train, method:str):
        self.X_train = X_train
        if method not in ['standard', 'minimax']:
            raise ValueError('method must be either "standard" or "minimax"')
        self.method = method

    def get_converted_data(self):
        
        method = self.method
        X_train = (X_train.self).copy()

        if method == 'standard':
            scaler = StandardScoreConverter(X_train)
            return scaler.get_converted_data()
        
        elif method == 'minimax':
            scaler = MinMaxConverter(X_train)
            return scaler.get_converted_data()

        else:
            raise TypeError('error')
    