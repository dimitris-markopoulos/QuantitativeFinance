import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
S0 = 100      # Initial stock price
mu = 0.05     # Drift
sigma = 0.2   # Volatility
T = 1.0       # Time horizon (1 year)
dt = 0.01     # Time step
N = int(T/dt) # Number of steps
n_paths = 5   # Number of GBM paths

# Simulate GBM
t = np.linspace(0, T, N)
paths = np.zeros((n_paths, N))
paths[:, 0] = S0

for i in range(1, N):
    Z = np.random.standard_normal(n_paths)
    paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * Z)

# Create animation
fig, ax = plt.subplots(figsize=(8, 5))
lines = [ax.plot([], [])[0] for _ in range(n_paths)]

def init():
    ax.set_xlim(0, T)
    ax.set_ylim(0.5 * S0, 2 * S0)
    ax.axis("off")  # Hides everything: ticks, borders, labels
    return lines

def update(frame):
    for line, path in zip(lines, paths):
        line.set_data(t[:frame], path[:frame])
    return lines

ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, blit=True)

# Save as GIF
ani.save("gbm_simulation.gif", writer="pillow", fps=30)
