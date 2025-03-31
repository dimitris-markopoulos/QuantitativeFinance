import numpy as np
import scipy.stats as stats

class BSM:
    model_name = "Black-Scholes-Merton"
    def __init__(self, S0, K, r, q, T, v):
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.v = v

        self.d1 = (np.log(S0 / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
        self.d2 = self.d1 - v * np.sqrt(T)
    
    def EuropeanCall(self):
        price = self.S0*np.exp(-self.q*self.T)*stats.norm.cdf(self.d1) - self.K*np.exp(-self.r*self.T)*stats.norm.cdf(self.d2)
        return price
    
    def EuropeanPut(self):
        price = self.K*np.exp(-self.r*self.T)*stats.norm.cdf(-self.d2) - self.S0*np.exp(-self.q*self.T)*stats.norm.cdf(-self.d1)
        return price
