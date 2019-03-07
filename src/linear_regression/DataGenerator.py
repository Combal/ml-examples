import numpy as np
import random
# don't remove
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

random.seed(1)


class DataGenerator:
    _np_data = []

    @staticmethod
    def _h(x1, x2):
        return 1 + x1 + x2

    @staticmethod
    def _generator():
        while True:
            x1 = random.uniform(-100, 100)
            x2 = x1 + random.uniform(-20, 20)
            yield x1, x2, DataGenerator._h(x1, x2)

    def generate(self, n):
        gen = self._generator()
        data = []
        for i in range(0, n):
            x1, x2, y = next(gen)
            data.append([x1, x2, y + random.uniform(-50, 50)])
        self._np_data = np.array(data, dtype=float)
        return self._np_data

    def plot(self):
        ax = plt.axes(projection='3d')

        # draw line
        x1 = np.linspace(-100, 100)
        x2 = np.linspace(-100, 100)
        ax.plot3D(x1, x2, self._h(x1, x2), 'b')

        # draw points
        ax.scatter3D(self._np_data[:, 0], self._np_data[:, 1], self._np_data[:, 2], 'ro')

        plt.show()

    def save_data(self):
        # pd.DataFrame(self._np_data).to_csv("path/to/file.csv")
        cwd = os.getcwd()
        path_to_data = os.path.join(os.path.realpath(cwd), 'data')
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        np.savetxt(os.path.join(path_to_data, 'linear_regression.csv'), self._np_data, delimiter=",")


if __name__ == '__main__':
    generator = DataGenerator()
    np_data = generator.generate(500)
    generator.plot()
