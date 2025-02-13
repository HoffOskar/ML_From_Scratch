import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(X_test, y_test, model): 
    """
    Plot the decision boundary for a trained classifier and true labels of X_test.

    Parameters:
        X_test (np.array): Shape (n_samples, 2), feature matrix (only 2D).
        y_test (np.array): Shape (n_samples,), corresponding labels.
        model (object): Trained classifier with a .predict() method.
    """
    ### Instantiate the figure
    fig, ax = plt.subplots()

    ### Create a meshgrid
    x0_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 500)
    x1_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 500)
    x0, x1 = np.meshgrid(x0_range, x1_range)

    ### Predict the classes on the meshgrid
    X_new = np.c_[x0.ravel(), x1.ravel()]  # Stack x0 and x1 as feature vectors
    y_pred = model.predict(X_new)
    zz = y_pred.reshape(x0.shape)

    ### Plot the decision boundary
    ax.contourf(x0, x1, zz, alpha=0.2)                        # Fill the decision boundary
    ax.contour(x0, x1, zz, colors='red', linewidths=0.4)      # Add decision boundary lines
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='black')      # Plot original data points

    ### Remove axis labels & ticks for clean visualization
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

def subplot_boundary(ax, X_test, y_test, model): 
    """
    Plot the decision boundary for a trained classifier and true labels of X_test on a given subplot.

    Parameters:
        ax (matplotlib.axes.Axes): Subplot axis to plot on.
        X_test (np.array): Shape (n_samples, 2), feature matrix (only 2D).
        y_test (np.array): Shape (n_samples,), corresponding labels.
        model (object): Trained classifier with a .predict() method.
    """
    ### Create a meshgrid
    x0_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 500)
    x1_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 500)
    x0, x1 = np.meshgrid(x0_range, x1_range)

    ### Predict the classes on the meshgrid
    X_new = np.c_[x0.ravel(), x1.ravel()]  # Stack x0 and x1 as feature vectors
    y_pred = model.predict(X_new)
    zz = y_pred.reshape(x0.shape)

    ### Plot the decision boundary
    ax.contourf(x0, x1, zz, alpha=0.2)                        # Fill the decision boundary
    ax.contour(x0, x1, zz, colors='red', linewidths=0.4)      # Add decision boundary lines
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='black')  # Plot original data points

    ### Remove axis labels & ticks for clean visualization
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
