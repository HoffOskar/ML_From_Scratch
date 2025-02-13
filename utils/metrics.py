### Imports
import numpy as np

def MSE(y_true, y_pred):
    """
    Calculate the mean squared error between two arrays.
    """
    return np.mean((y_true - y_pred) ** 2)

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy between two arrays.
    """
    return np.mean(y_true == y_pred)
