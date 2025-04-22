import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Pipeline():
    def __init__(self, X_tr,y_tr,X_ts,y_ts):
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.y_tr = y_tr
        self.y_ts = y_ts
        
