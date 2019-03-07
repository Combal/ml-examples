import numpy as np

from .DataGenerator import DataGenerator
from .feature_normalize import feature_normalize
from .gradient_decent import gradient_decent
from .compute_cost import plot_cost

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # step 1 - generate random data and plot
    generator = DataGenerator()
    data = generator.generate(500)
    generator.plot()
    # generator.save_data()

    # step 2 - feature normalization
    m, n = data.shape
    # print("data shape {}".format(data.shape))
    X = data[:, 0:n - 1]
    y = data[:, n - 1]
    # print("X shape {}".format(X.shape))
    # print("y size {}".format(y.size))
    # X, mu, sigma = feature_normalize(X)
    # print(mu, sigma)
    X = np.insert(X, 0, np.ones(m), axis=1)
    # print(X)

    # step 3 - gradient decent
    alpha = 0.0002
    num_iters = 10
    _, n = X.shape
    theta = np.zeros(n)
    theta, J_history = gradient_decent(X, y, theta, alpha, num_iters)
    # print('J_history: {}'.format(J_history))
    print('computed theta values:')
    print(theta)
    plot_cost(J_history)

    ax = plt.axes(projection='3d')

    # draw line
    x1 = np.linspace(-100, 100)
    x2 = np.linspace(-100, 100)
    ax.plot3D(x1, x2, theta[0] + theta[1] * x1 + theta[2] * x2, 'r')
    # ax.plot3D(x1, x2, (1+x1+x2), 'b')
    data = generator.get_data()
    # draw points
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], 'ro')

    plt.show()

