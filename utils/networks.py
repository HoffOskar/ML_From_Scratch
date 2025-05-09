### Imports
import numpy as np


### Class Definition
class MLP:
    """
    Multi-Layer Perceptron (MLP) for regression with a single hidden layer, tanh activation function, and gradient decent.

    Attributes:
        hidden_dim (int): Number of neurons in the hidden layer
        learning_rate (float): Step size for the gradient descent
        max_iter (int): Maximum number of iterations
        alpha (float): Regularization parameter
        W1 (np.array): Weights for the hidden layer (n_hidden, n_features)
        W2 (np.array): Weights for the output layer (n_outputs, n_hidden)
        b1 (np.array): Biases for the hidden layer (n_hidden, 1)
        b2 (np.array): Biases for the output layer (n_outputs, 1)
        loss (np.array): Loss function values during training (max_iter,)
    """

    def __init__(
        self,
        hidden_dim=10,
        learning_rate=0.05,
        max_iter=3000,
        alpha=0.001,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, Y, cold_start=True):
        """
        Fit the MLP model to the data.

        Parameters:
            X (np.array): Features (n_samples, n_features)
            Y (np.array): Labels (n_samples, n_outputs)
            cold_start (bool): Initialize weights randomly if True, else use previous weights
            random_state (int): Seed for the random number generator
        """
        ### Hyperparameters
        h = self.hidden_dim  # Number of neurons in the hidden layer
        alpha = self.alpha  # Regularization parameter
        max_it = self.max_iter  # Maximum number of iterations
        step_size = self.learning_rate  # Step size

        ### Data format
        d = X.shape[1]  # Number of features
        n = X.shape[0]  # Number of samples
        o = Y.shape[1]  # Number of outputs

        ### Initialize random weights and biases
        if cold_start:
            np.random.seed(self.random_state)

            ### Initialize weights with Xavier initialization
            limit_W1 = np.sqrt(6 / (d + h))
            self.W1 = np.random.uniform(-limit_W1, limit_W1, size=(h, d))

            limit_W2 = np.sqrt(6 / (h + o))
            self.W2 = np.random.uniform(-limit_W2, limit_W2, size=(o, h))

            ### Initialize biases to zero
            self.b1 = np.zeros((h, 1))
            self.b2 = np.zeros((o, 1))

        ### Training loop
        iter = 0  # Iteration counter
        losses = np.zeros(max_it)  # Container for the regularized loss function
        MSEs = np.zeros(max_it)  # Container for the mean squared error

        while iter < max_it:
            ### Predict Labels (Forward Pass)
            Z = self.W1 @ X.T + self.b1  # Hidden layer linear transformation
            H = np.tanh(Z)  # Activation function
            Y_pred = self.W2 @ H + self.b2  # Output layer
            Y_pred = Y_pred.T  # Transpose back to (n, o)

            ### Compute Gradients (Backpropagation)

            ### Compute error
            delta_2 = (Y_pred - Y) * (2 / n)  # (n, o)

            ### Gradients for W2 and b2
            grad_W2 = delta_2.T @ H.T + alpha * self.W2  # (o, h)
            grad_b2 = np.sum(delta_2, axis=0, keepdims=True) + alpha * self.b2  # (o, 1)

            ### Backprop to hidden layer
            delta_1 = (self.W2.T @ delta_2.T) * (1 - H**2)  # (h, n)

            ### Gradients for W1 and b1
            grad_W1 = delta_1 @ X + alpha * self.W1  # (h, d)
            grad_b1 = np.sum(delta_1, axis=1, keepdims=True) + alpha * self.b1  # (h, 1)

            ### Update Parameters (Gradient Descent)
            self.W1 -= step_size * grad_W1
            self.b1 -= step_size * grad_b1
            self.W2 -= step_size * grad_W2
            self.b2 -= step_size * grad_b2

            ### Compute Loss
            losses[iter] = ((Y - Y_pred) ** 2).sum() / (2 * n) + alpha * (
                (self.b1**2).sum()
                + (self.b2**2).sum()
                + (self.W1**2).sum()
                + (self.W2**2).sum()
            ) / 2

            MSEs[iter] = ((Y - Y_pred) ** 2).sum() / (2 * n)

            ### Increment iteration counter
            iter += 1

        ### Save loss function values
        self.loss = losses[:iter]
        self.mse = MSEs[:iter]

    def predict(self, X):
        """
        Predict labels for new data.

        Parameters:
            X (np.array): Features (n_samples, n_features)

        Returns:
            y_pred (np.array): Predicted labels (n_samples, n_outputs)
        """
        ### Predict Labels (Forward Pass)
        Z = self.W1 @ X.T + self.b1  # Hidden layer linear transformation
        H = np.tanh(Z)  # Activation function
        Y_pred = self.W2 @ H + self.b2  # Output layer
        Y_pred = Y_pred.T  # Transpose back to (n, o)
        return Y_pred
