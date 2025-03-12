### Imports
import numpy as np


### Classes
class RidgeReg:
    """
    Ridge regression model.

    Attributes:
        alpha (float): The regularization parameter.
        beta (np.array): The coefficients of the model.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.beta = None

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters:
            X (np.array): The features.
            y (np.array): The targets.

        Side effects:
            Updates self.beta with the fitted coefficients
        """
        ### Add a column of ones to X to account for the intercept
        X = np.c_[np.ones(X.shape[0]), X]

        ### Compute the coefficients
        beta = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ y)

        ### Store the coefficients
        self.beta = beta

    def predict(self, X):
        """
        Predict the targets for new data.

        Parameters:
            X (np.array): The features.

        Returns:
            y (np.array): The predicted targets.
        """
        ### Add a column of ones to X to account for the intercept
        X = np.c_[np.ones(X.shape[0]), X]

        ### Predictions
        y = X @ self.beta
        return y
