import numpy as np


def feature_normalize(X):
    X_norm = X
    _, n = X.shape
    mu = np.zeros(n)
    sigma = np.zeros(n)

    for i in range(0, n):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma
