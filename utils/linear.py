### Imports
import numpy as np

### Classes


class LogReg:
    """
    Logistic regression model with L2 regularization.

    Attributes:
        alpha (float): Regularization strength
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Number of iterations for gradient descent
        beta (np.array): Coefficients (n_features + 1,)
        proba (np.array): Predicted probabilities (n_samples,)
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, max_iter=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta = None
        self.proba = None

    def fit(self, X, y):
        """
        Compute the coefficients beta iteratively using gradient descent.

        Parameters:
            X (np.array): Features (n_samples, n_features)
            y (np.array): Binary labels (n_samples,)

        Side effects:
            Updates self.beta with fitted coefficients.
        """
        ### Add intercept column
        X = np.c_[np.ones(X.shape[0]), X]
        n, d = X.shape

        ### Initialize beta (coefficients)
        self.beta = np.zeros(d)

        ### Gradient descent loop
        for _ in range(self.max_iter):
            y_pred = self._sigmoid(X @ self.beta)  # Predicted probabilities
            gradient = X.T @ (y_pred - y) + 2 * self.alpha * np.r_[0, self.beta[1:]]
            self.beta -= self.learning_rate * gradient

    def predict(self, X):
        """
        Predict binary labels for samples in X.

        Parameters:
            X (np.array): Features (n_samples, n_features)

        Side effects:
            Updates self.proba with predicted probabilities.
        """
        ### Store probabilities
        self.proba = self._predict_proba(X)

        ### Return binary predictions
        return (self.proba >= 0.5).astype(int)

    ### Helper functions

    def _sigmoid(self, z):
        """
        Compute the sigmoid function for z.
        """
        return 1 / (1 + np.exp(-z))

    def _predict_proba(self, X):
        """
        Compute predicted probabilities for samples in X.

        Parameters:
            X (np.array): Features (n_samples, n_features)

        Returns:
            np.array: Predicted probabilities (n_samples,)
        """

        ### Add intercept column
        X = np.c_[np.ones(X.shape[0]), X]

        ### Predict probabilities
        return self._sigmoid(X @ self.beta)


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
