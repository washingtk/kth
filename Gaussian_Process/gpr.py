import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import OneHotEncoder
# import some data to play with


class GP:

    def __init__(self, test_size, kernel):
        self.test_size = test_size
        self.x_train, self.x_test, self.y_train, self.y_test = self.set_size()
        if kernel == 1:
            self.kernel = 1.0 * RBF([1.0])
        elif kernel == 2:
            self.kernel = DotProduct(sigma_0=1.0)
        else:
            self.kernel = 1.0 * RBF([1.0])


    def set_size(self):
        digits = datasets.load_digits()

        x = digits.data
        y = np.array(digits.target, dtype=int)
        y =
        _x_train, _x_test, _y_train, _y_test = train_test_split(x, y, test_size=self.test_size, random_state=121)

        return _x_train, _x_test, _y_train, _y_test

    def error_rate(self):
        gpr = GaussianProcessRegressor(kernel=self.kernel, random_state=121).fit(self.x_train, self.y_train)
        score = gpr.score(self.x_train, self.y_train)
        yp_test = np.argmax(gpr.predict(self.x_test))
        test_error_rate = np.mean(np.not_equal(yp_test, self.y_test))

        return score, test_error_rate

a = GP(180, 1)
a = a.error_rate()
print(a[0], a[1])