import numpy as np


def compute_cost(X, y, theta):
    m = y.size
    x_theta = np.dot(X, theta)
    errors = x_theta - y
    J = 1 / (2 * m) * (np.transpose(errors) @ errors)
    return J
