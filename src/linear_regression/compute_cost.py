import numpy as np


def compute_cost(X, y, theta):
    m = y.size
    errors = (X @ theta) - y
    J = 1 / (2 * m) * (np.transpose(errors) @ errors)
    return J
