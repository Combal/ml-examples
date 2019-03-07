from .DataGenerator import DataGenerator
from .featureNormalize import featureNormalize

if __name__ == '__main__':
    # step 1 - data generator and plot
    generator = DataGenerator()
    data = generator.generate(500)
    generator.plot()
    generator.saveData()

    # step 2 - feature normalization
    n, m = data.shape
    print("data shape {}".format(data.shape))
    X = data[:, 0:m - 1]
    y = data[:, m - 1]
    print("X shape {}".format(X.shape))
    print("y size {}".format(y.size))

    X_norm, mu, sigma = featureNormalize(X)
    # print(mu, sigma)
