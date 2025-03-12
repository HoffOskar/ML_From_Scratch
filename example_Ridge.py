### Imports

import sklearn.datasets as skd
from sklearn.model_selection import train_test_split

from utils.linear import RidgeReg
from utils.metrics import MSE

### Data
dataset = skd.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    dataset["data"], dataset["target"], test_size=0.2, random_state=2
)

### Fit the model
model = RidgeReg(alpha=0.5)
model.fit(X_train, y_train)

### Print errors
print(f"Training error: {MSE(model.predict(X_train), y_train):.0f}")
print(f"Test error: {MSE(model.predict(X_test), y_test):.0f}")
