### Imports
import matplotlib.pyplot as plt
import numpy as np

from utils.networks import MLP

### Samples
n = 100

### Noise
sig = 0.3

### Generate data
x = np.random.uniform(0, 1, n)
y = np.sin(3 * np.pi * x) + np.random.normal(0, sig, n)

### Instantiate model
model = MLP(max_iter=5_000)

### Fit model
model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

### Plot
X_plot = np.linspace(0, 1, 1000)[:, None]
Y_plot_pred = model.predict(X_plot)
Y_plot_0 = np.sin(3 * np.pi * X_plot)

# Plot the noisy and predicted data
plt.scatter(x, y, label="Training Data")
plt.plot(X_plot, Y_plot_0, label="Ground Truth")
plt.plot(X_plot, Y_plot_pred, label="Predicted")
plt.legend()
plt.show()
