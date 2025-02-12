### Imports
import numpy as np
from utils.tree import RegressionTree
from utils.metrics import MSE

### Function for data generation
def f(x, sigma):     
    return x**2+ np.random.normal(0, sigma, len(x))

### Data
np.random.seed(10)
X = np.linspace(-1, 1, 50).reshape(-1, 1)
y = f(X, 0.2)

### Hyperparameters
max_depth = 3
min_samples_split = 2

### Instantiate regression tree
tree = RegressionTree(max_depth=max_depth, min_samples_split=min_samples_split)

### Fit
tree.fit(X, y)

### Predict
y_pred = tree.predict(X)

### Calculate the training MSE
print(f'Training MSE: {round(MSE(y, y_pred), 3)}')