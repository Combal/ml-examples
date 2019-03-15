import numpy as np

from .data_generator import DataGenerator
from .feature_normalize import feature_normalize
from .gradient_descent import gradient_descent
from .compute_cost import plot_cost

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random

random.seed(1)

alpha = 0.0002
num_iters = 20
x_range = (-100, 100)


# step 1 - generate random data and plot
generator = DataGenerator(theta=[3, -2, 10], x_range=x_range)
data = generator.generate(500)
# generator.plot()
# generator.save_data()


# step 2 - feature normalization
m, n = data.shape
# print("data shape {}".format(data.shape))
X = data[:, 0:n - 1]
y = data[:, n - 1]
# print("X shape {}".format(X.shape))
# print("y size {}".format(y.size))
# X, mu, sigma = feature_normalize(X)
X = np.insert(X, 0, np.ones(m), axis=1)


# step 3 - gradient decent
_, n = X.shape
theta = np.random.rand(n)
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
print('learned theta: {}'.format(theta))
print('cost: {}'.format(J_history[-1]))
plot_cost(J_history)


# step 4 - plot prediction
generator.plot(theta, title='Prediction')
