### Imports
import numpy as np


class KNeighborsClassifier:
    """
    A simple K-Nearest Neighbors class.

    Attributes:
        k (int): Number of neighbors to consider
        dist_matrix (np.array): Unsorted distance matrix between training and test data
        X_train (np.array): Training data
        y_train (np.array): Training labels
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        """
        Store training data following the syntax of scikit-learn.

        Parameters:
            X_train (np.array): Training data
            y_train (np.array): Training labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict labels based on K-nearest neighbors (majority vote).

        Parameters:
            X_train (np.array): Training data
            y_train (np.array): Training labels
            X_test (np.array): Test data

        Returns:
            y_pred (np.array): NumPy array with the predicted labels for X_test.
        """

        ### Get the number of neighbors
        k = self.k
        X_train = self.X_train
        y_train = self.y_train

        ### Calculate all Euclidean distances between training and test data
        dist_np = self._distance_matrix(X_train, X_test)

        ### Get the indices of the k nearest neighbors
        k_idx = np.argpartition(dist_np, k, axis=1)[:, :k]

        ### Get the k nearest labels
        k_labels = y_train[k_idx]

        ### Get the most common label (majority vote)
        y_pred = np.array([np.bincount(labels).argmax() for labels in k_labels])

        ### Update class attribute
        self.dist_matrix = dist_np

        return y_pred

    ### Helper function

    def _distance_matrix(self, X_train, X_test):
        """
        Calculate a Euclidean distance matrix between two data sets.

        Parameters:
            X_train (np.array): 2D NumPy array with shape (n_train, n_features)
            X_test (np.array): 2D NumPy array with shape (n_test, n_features)

        Returns:
            distances (np.array):  2D NumPy array with shape (n_test, n_train)
        """
        return np.sqrt(((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2).sum(axis=2))
