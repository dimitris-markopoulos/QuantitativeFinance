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
        
