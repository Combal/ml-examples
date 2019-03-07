from .DataGenerator import DataGenerator
from .feature_normalize import feature_normalize


if __name__ == '__main__':

    # step 1 - data generator and plot
    generator = DataGenerator()
    data = generator.generate(500)
    generator.plot()
    # generator.save_data()

    # step 2 - feature normalization
    n, m = data.shape
    X = data[:, 0:m - 1]
    y = data[:, m - 1]
    # print("data shape {}".format(data.shape))
    # print("X shape {}".format(X.shape))
    # print("y size {}".format(y.size))

    X_norm, mu, sigma = feature_normalize(X)
    # print(mu, sigma)
