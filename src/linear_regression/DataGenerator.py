import numpy as np
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def h(x1, x2):
    return 1 + x1 + x2


def generator():
    while True:
        x1 = random.uniform(-100, 100)
        x2 = x1 + random.uniform(-20, 20)
        yield x1, x2, h(x1, x2)


def generate(n):
    gen = generator()
    data = []
    for i in range(0, n):
        x1, x2, y = next(gen)
        data.append([x1, x2, y + random.uniform(-50, 50)])
    return np.array(data, dtype=float)


if __name__ == '__main__':
    np_data = generate(500)
    # print(np_data)
    x1 = np.linspace(-100, 100)
    x2 = np.linspace(-100, 100)
    ax = plt.axes(projection='3d')
    # plt.plot(np_data[:, 0], np_data[:, 1], 'ro', ms=2)
    # plt.plot(x, h(x), 'b')
    ax.scatter3D(np_data[:, 0], np_data[:, 1], np_data[:, 2], 'ro')
    ax.plot3D(x1, x2, h(x1, x2), 'b')
    plt.show()
