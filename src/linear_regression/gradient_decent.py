import numpy as np

from .compute_cost import compute_cost


def gradient_decent(X, y, theta, alpha, num_iters):
    m, n = X.shape
    J_history = []
    for i in range(0, num_iters):
        errors = (X @ theta) - y
        theta = theta - alpha / m * (X.T @ errors)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history
