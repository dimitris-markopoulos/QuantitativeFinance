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

    @property
    def EuropeanCall(self):
        price = self.S0*np.exp(-self.q*self.T)*stats.norm.cdf(self.d1) - self.K*np.exp(-self.r*self.T)*stats.norm.cdf(self.d2)
        return price
    
    @property
    def EuropeanPut(self):
        price = self.K*np.exp(-self.r*self.T)*stats.norm.cdf(-self.d2) - self.S0*np.exp(-self.q*self.T)*stats.norm.cdf(-self.d1)
        return price
    
    @classmethod
    def days_to_years(cls, S0, K, r, q, T_days, v):
        T_years = T_days / 365
        return cls(S0, K, r, q, T_years, v)

class BSMGreeks(BSM):
    def __init__(self, S0, K, r, q, T, v):
        super().__init__(S0, K, r, q, T, v)  # Inherit everything

    @property
    def delta_call(self):
        return np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)

    @property
    def delta_put(self):
        return np.exp(-self.q * self.T) * (stats.norm.cdf(self.d1) - 1)

    @property
    def gamma(self):
        numerator = stats.norm.pdf(self.d1) * np.exp(-self.q * self.T)
        denominator = self.S0 * self.v * np.sqrt(self.T)
        return numerator / denominator

    @property
    def vega(self):
        return self.S0 * stats.norm.pdf(self.d1) * np.sqrt(self.T) * np.exp(-self.q * self.T)
