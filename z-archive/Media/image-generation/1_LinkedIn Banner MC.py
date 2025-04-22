import numpy as np
import matplotlib.pyplot as plt

name = 'Dimitris Markopoulos' #INSERT NAME HERE

S0 = 100   # Initial stock price
K = 100    # Strike price
T = 1      # Time to maturity (1 year)
r = 0.05   # Risk-free rate
q = 0.02   # Dividend yield
v = 0.3    # Volatility
n = 252    # Number of time steps
M = 10000  # Number of simulations
dt = T / n
paths = []

for _ in range(M):
    path = [S0]
    for _ in range(n):
        z = np.random.randn()
        S_t = path[-1] * np.exp((r - q - 0.5 * v**2) * dt + v * np.sqrt(dt) * z)
        path.append(S_t)
    paths.append(path)

paths = np.array(paths)

dpi = 500 # Increased DPI for high definition
fig_width = 1584 / 100
fig_height = 396 / 100
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)  

for i in range(200):
    ax.plot(paths[i], alpha=0.8, linewidth=1)

# Add mathematical annotations, positioned higher to avoid overlapping with paths
ax.text(10, S0 + 80, r"$S_t = S_0 e^{\left(r - q - \frac{1}{2} \sigma^2\right)t + \sigma W_t}$", 
        fontsize=20, color="black", fontweight="bold")
ax.text(10, S0 + 60, r"$A = \frac{1}{n} \sum_{i=1}^{n} S_i$", 
        fontsize=10, color="black")
ax.text(240, 30, f"{name}", fontsize=5, color="black", ha="right", va="bottom", fontstyle="italic")
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Save as a high-resolution image for LinkedIn banner
plt.savefig("linkedin_banner.png", dpi=dpi, bbox_inches='tight', transparent=False)
plt.show()