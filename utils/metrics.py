### Imports
import numpy as np


### Mean Squared Error
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
