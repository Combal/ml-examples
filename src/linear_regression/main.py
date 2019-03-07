import numpy as np

from .DataGenerator import DataGenerator
from .feature_normalize import feature_normalize
from .gradient_decent import gradient_decent

if __name__ == '__main__':
    # step 1 - generate random data and plot
    generator = DataGenerator()
    data = generator.generate(500)
    generator.plot()
    generator.save_data()

    # step 2 - feature normalization
    m, n = data.shape
    # print("data shape {}".format(data.shape))
    X = data[:, 0:n - 1]
    y = data[:, n - 1]
    # print("X shape {}".format(X.shape))
    # print("y size {}".format(y.size))
    X, mu, sigma = feature_normalize(X)
    # print(mu, sigma)
    X = np.insert(X, 0, np.ones(m), axis=1)
    # print(X)

    # step 3 - gradient decent
    alpha = 0.01
    num_iters = 50
    _, n = X.shape
    theta = np.zeros(n)
    theta, J_history = gradient_decent(X, y, theta, alpha, num_iters)
    print('J_history: {}'.format(J_history))
    print('computed theta values:')
    print(theta)
