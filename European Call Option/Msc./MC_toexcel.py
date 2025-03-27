import pandas as pd
import numpy as np
import matplotlib.pyplot

S0, v, r, q, T = 50, 0.30, 0.05, 0.00, 1
n = 250 #time steps

def paths_dataframe(M, S0 = 50, v = 0.30, r = 0.05, q = 0.00, T = 1, n = 250):
    dt = T/n
    paths_dict = {i:[] for i in range(M)}
    for _ in range(M):
        paths = [S0]
        for i in range(n - 1):
            Z = np.random.randn()
            S = paths[-1] * np.exp((r - q - (v**2)/2)*dt + v * Z * np.sqrt(dt))
            paths.append(S)
        paths_dict[_].extend(paths)

    df = pd.DataFrame(paths_dict).T
    df.columns = [f't{i}' for i in range(n)]
    return df


M = 20000 #number of paths
df = paths_dataframe(M)
df.to_excel('20k_MC.xlsx', index=False)