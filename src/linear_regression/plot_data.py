from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot3d(data):

    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], 'ro', s=2)
    ax.set_xlabel('X1 axis')
    ax.set_ylabel('X2 axis')
    ax.set_zlabel('Y axis')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        plot3d(np.loadtxt(sys.argv[1], delimiter=','))
    else:
        print('Error: argument missing')
