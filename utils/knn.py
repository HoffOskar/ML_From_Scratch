### Imports
import numpy as np


class KMeans:
    """
    A simple K-means class for clustering.

    Attributes:
        n_clusters (int): Number of clusters
        max_iter (int): Maximum number of iterations
        seed (int): Random seed
        centroids (np.array): Centroids of the clusters
    """

    def __init__(self, n_clusters=3, max_iter=100, seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def fit_predict(self, X):
        """
        Compute centroids and assign labels to samples.

        Parameters:
            X (np.array): Samples with the shape (n_samples, n_features)

        Returns:
            y_pred (np.array): 1D NumPy array with the cluster labels

        Side effects:
            self.centroids (np.array): Cluster centroids with shape (n_clusters, n_features) are updated.
        """

        ### Subsetting random nc data points as initial centroids
        np.random.seed(self.seed)
        random_idx = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        centroids = X[random_idx]

        ### Loop parameters
        converged = False
        i = 1

        ### Loop until convergence
        while not converged:
            ### Distance matrix
            dist_np = self._distance_matrix(X, centroids)

            ### Assigning clusters labels to samples
            y_pred = np.argmin(dist_np, axis=0)

            ### Calculate new centroids
            new_centroids = np.array(
                [X[np.where(y_pred == k)].mean(axis=0) for k in range(self.n_clusters)]
            )

            ### Check if the centroids have changed
            if np.all(centroids == new_centroids):
                ### Stop if the centroids have not changed
                converged = True

                ### Update the class centroids
                self.centroids = centroids

                ### Return the centroids and the data labels
                return y_pred

            ### Stop after max_iter iterations
            elif i >= self.max_iter:
                print(f"Stopped after {i} iterations")
                converged = True

            ### Set up for next iteration
            else:
                centroids = new_centroids
                i += 1

    def _distance_matrix(self, X_train, X_test):
        """
        Calculate a Euclidean distance matrix between two data sets.

        Parameters:
            X_train (np.array): 2D NumPy array with shape (n_train, n_features)
            X_test (np.array): 2D NumPy array with shape (n_test, n_features)

        Returns:
            distances (np.array):  2D NumPy array with shape (n_test, n_train)
        """
        return np.sqrt(
            ((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2).sum(axis=2)
        )


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
        return np.sqrt(
            ((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
