import numpy as np
import random
# don't remove
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os


class DataGenerator:
    _np_data = []
    _x_deviation = 20    # x2-ის გადახრა x1-ისგან

    def __init__(self, theta, x_range=(-100, 100)):
        self._theta = np.array(theta)
        self._x_range = x_range
        print('desired theta: {}'.format(theta))

    def h(self, x, theta=None):
        x = np.array(x)
        theta = np.array(theta) if theta is not None else self._theta
        return np.dot(theta, x)         # same as: t0 + t1 * x1 + t2 * x2

    def _generator(self):
        a, b = self._x_range

        while True:
            x1 = random.uniform(a, b)
            x2 = x1 + random.uniform(-self._x_deviation, self._x_deviation)
            yield x1, x2, self.h([1, x1, x2])

    def generate(self, n):
        gen = self._generator()
        data = []
        for i in range(0, n):
            x1, x2, y = next(gen)
            data.append([x1, x2, y + self._get_y_deviation()])
        self._np_data = np.array(data, dtype=float)
        return self.get_data()

    def get_data(self):
        return np.copy(self._np_data)

    def _get_y_deviation(self):
        """ y-ის გადახრა რეალური მნიშვნელობისგან """
        _, b = self._x_range
        d = self.h([1, b, b]) * self._x_deviation / b
        return random.uniform(-d, d)

    def plot(self, theta=None, title=None):
        theta = theta if theta is not None else self._theta
        title = title if title is not None else 'Generated Data'
        ax = plt.axes(projection='3d')
        plt.title(title)
        a, b = self._x_range
        # draw line
        x1 = np.linspace(a, b)
        x2 = np.linspace(a, b)
        ax.plot3D(x1, x2, self.h([1, x1, x2], theta), 'b')

        # draw points
        ax.scatter3D(self._np_data[:, 0], self._np_data[:, 1], self._np_data[:, 2], 'ro', s=2)
        ax.set_xlabel('X1 axis')
        ax.set_ylabel('X2 axis')
        ax.set_zlabel('Y axis')

        plt.show()

    def save_data(self):
        # pd.DataFrame(self._np_data).to_csv("path/to/file.csv")
        cwd = os.getcwd()
        path_to_data = os.path.join(os.path.realpath(cwd), 'data')
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        np.savetxt(os.path.join(path_to_data, 'linear_regression.csv'), self._np_data, delimiter=",")


if __name__ == '__main__':
    generator = DataGenerator(theta=[
        random.uniform(-100, 100),
        random.uniform(-100, 100),
        random.uniform(-100, 100)])
    np_data = generator.generate(500)
    generator.plot()
