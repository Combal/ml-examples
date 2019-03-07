import numpy as np


# noinspection PyPep8Naming
def feature_normalize(X):
    X_norm = X
    _, m = X.shape
    mu = np.zeros(m)
    sigma = np.zeros(m)

    for i in range(0, m):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma
