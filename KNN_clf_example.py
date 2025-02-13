### Imports
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from utils.knn import KNeighborsClassifier
from utils.metrics import accuracy

### Settings
np.random.seed(0)

### Generating Data
X, y = make_classification(
    n_samples=100,  # 100 samples
    n_features=2,  # 2 features
    n_redundant=0,  # No redundant features - all features are informative
    n_informative=2,  # 2 informative features - all features are informative
    random_state=3,
)  # Random state for reproducibility

### Splitting Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3
)

### Instantiate custom KNN Classifier
knn = KNeighborsClassifier(k=3)

### Fit
knn.fit(X_train, y_train)

### Predict
y_pred = knn.predict(X_test)

### Accuracy
print(f"Test accuracy: {accuracy(y_test, y_pred):.2f}")
