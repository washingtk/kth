import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process.kernels import DotProduct
from itertools import product


"""

Use Gaussian Process classification to distinguish a MNIST data and
compare a result with two kernel and different test size

"""


class GP:

    def __init__(self, test_size, kernel):
        self.test_size = test_size
        self.kernel = kernel
        self.x_train, self.x_test, self.y_train, self.y_test = self.set_size()

    def set_size(self):
        digits = datasets.load_digits()

        x = digits.data[0:599, :]
        y = np.array(digits.target, dtype=int)[0:599]
        _x_train, _x_test, _y_train, _y_test = train_test_split(x, y, test_size=self.test_size, random_state=121)

        return _x_train, _x_test, _y_train, _y_test

    def error_rate(self):
        if self.kernel == "RBF":
            gpc = GaussianProcessClassifier(kernel=1.0 * RBF([1.0])).fit(self.x_train, self.y_train)
        elif self.kernel == "DP":
            gpc = GaussianProcessClassifier(kernel=DotProduct(sigma_0=1.0)).fit(self.x_train, self.y_train)
        else:
            print("Error")
        yp_train = gpc.predict(self.x_train)
        train_error_rate = np.mean(np.not_equal(yp_train, self.y_train))
        yp_test = gpc.predict(self.x_test)
        test_error_rate = np.mean(np.not_equal(yp_test, self.y_test))

        return train_error_rate, test_error_rate


test = [10, 20, 40, 70, 120]
kernel = ["RBF", "DP"]

for test_size, kernel in product(test, kernel):
    gp = GP(test_size=test_size, kernel=kernel)
    res = gp.error_rate()

    print("test size:{0}    Kernel:{1}".format(test_size, kernel))
    print('Train error rate:{0} \n Test error rate:{1}'.format(res[0], res[1]))