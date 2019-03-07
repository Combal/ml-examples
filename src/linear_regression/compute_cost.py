import numpy as np
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    m = y.size
    errors = (X @ theta) - y
    J = 1 / (2 * m) * (np.transpose(errors) @ errors)
    return J


def plot_cost(J_history):
    plt.plot(range(0, len(J_history)), J_history, 'r')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.show()
