### Imports
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from utils.knn import KMeans

### Generate data
X, y = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=1, random_state=2)

### Instantiate KMeans
k_means = KMeans(n_clusters=3)

### Predict clusters
y_pred = k_means.fit_predict(X)

### Plot clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.scatter(k_means.centroids[:, 0], k_means.centroids[:, 1], s=200, c='red', marker='X')
plt.title(f'KMeans Clustering')
plt.show()