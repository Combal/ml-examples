import numpy as np
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class DataGenerator:

    @staticmethod
    def h(x1, x2):
        return 1 + x1 + x2

    @staticmethod
    def _generator():
        while True:
            x1 = random.uniform(-100, 100)
            x2 = x1 + random.uniform(-20, 20)
            yield x1, x2, DataGenerator.h(x1, x2)

    def generate(self, n):
        gen = self._generator()
        data = []
        for i in range(0, n):
            x1, x2, y = next(gen)
            data.append([x1, x2, y + random.uniform(-50, 50)])
        return np.array(data, dtype=float)


if __name__ == '__main__':
    generator = DataGenerator()

    np_data = generator.generate(500)
    # print(np_data)
    x_1 = np.linspace(-100, 100)
    x_2 = np.linspace(-100, 100)
    ax = plt.axes(projection='3d')
    # plt.plot(np_data[:, 0], np_data[:, 1], 'ro', ms=2)
    # plt.plot(x, h(x), 'b')
    ax.scatter3D(np_data[:, 0], np_data[:, 1], np_data[:, 2], 'ro') # draw random  generated points
    ax.plot3D(x_1, x_2, generator.h(x_1, x_2), 'b') # draw line
    plt.show()
