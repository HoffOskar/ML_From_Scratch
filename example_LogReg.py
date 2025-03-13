### Imports

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.linear import LogReg

### Data
X = np.loadtxt("Classification/data/X_morgan_2048.csv", delimiter=",")
y_df = pd.read_csv("Classification/data/y.csv")

### Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_df["active"].to_numpy(), test_size=0.2, random_state=2
)

model = LogReg(alpha=0.1, learning_rate=0.01, max_iter=100)
model.fit(X_train, y_train)

print(f"Training accuracy: {accuracy_score(model.predict(X_train), y_train):.2f}")
print(f"Test accuracy: {accuracy_score(model.predict(X_test), y_test):.2f}")
print("Hyperparameters:")
print(f"alpha: {model.alpha}")
print(f"learning rate: {model.learning_rate}")
print(f"number of iterations: {model.max_iter}")
