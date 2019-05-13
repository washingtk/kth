from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping
from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence, plot_objective
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # to avoid an error with MacOS


# set up a class having all necessary stuff
class CNN:

    def __init__(self, test_size, filter1, nodes):
        self.test_size = test_size
        self.filter1 = filter1
        self.nodes = nodes
        self.x_train, self.x_test, self.y_train, self.y_test = self.datasets()
        self.cnn_model = self.firstmodel()

    def datasets(self):
        digits = datasets.load_digits()
        # normalization to speed up calculation
        x = np.asarray(digits.data, "float32")
        x /= 16
        y = np_utils.to_categorical(digits.target, 10)
        _x_train, _x_test, _y_train, _y_test = train_test_split(x, y, test_size=self.test_size, random_state=21)

        return _x_train, _x_test, _y_train, _y_test

    def firstmodel(self):
        model = Sequential()
        model.add(Conv2D(self.filter1, kernel_size=(3,3), strides=(1,1), padding='same',
                         activation="relu", input_shape=(8, 8, 1),))
        model.add(MaxPool2D(2, 2))
        model.add(Flatten())
        model.add(Dense(self.nodes, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

        return model

    def fit1(self):
        x_train = self.x_train.reshape(self.x_train.shape[0], 8, 8, 1)
        early_stop = EarlyStopping(patience=3, verbose=1)  # stop learning when stop decreasing value of loss/accuracy?
        self.cnn_model.fit(x_train, self.y_train, batch_size=10, epochs=15,
                           verbose=0, validation_split=0.1, callbacks=[early_stop])

    def evaluation1(self):
        x_test = self.x_test.reshape(self.x_test.shape[0], 8, 8, 1)
        self.fit1()
        evaluation1 = self.cnn_model.evaluate(x_test, self.y_test, verbose=0)

        return evaluation1



class NN(CNN):

    def __init__(self, test_size, neurons):
        self.test_size = test_size
        self.neurons = neurons
        self.x_train, self.x_test, self.y_train, self.y_test = self.datasets()
        self.nn_model = self.secondmodel()

    def secondmodel(self):
        model = Sequential()
        model.add(Dense(self.neurons, activation='relu', input_shape=(64,)))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

        return model

    def fit2(self):
        early_stop = EarlyStopping(patience=3, verbose=1)
        self.nn_model.fit(self.x_train, self.y_train, batch_size=10, epochs=15,
                          verbose=1, validation_split=0.1, callbacks=[early_stop])

    def evaluation2(self):
        self.fit2()
        evaluation2 = self.nn_model.evaluate(self.x_test, self.y_test, verbose=0)

        return evaluation2


# function to run
def run1(test_size, filter1, nodes):
    cnn = CNN(test_size=test_size, filter1=filter1, nodes=nodes)
    cnn_eval = cnn.evaluation1()

    return cnn_eval


def run2(test_size, neurons):
    nn = NN(test_size=test_size, neurons=neurons)
    nn_eval = nn.evaluation2()

    return nn_eval


# function to minimize/optimize
def f1(x):
    print(x)
    eval = run1(test_size=x[0], filter1=x[1], nodes=x[2])
    print("Loss:{0}     Accuracy:{1}".format(eval[0], eval[1]))

    return 1 - eval[1]


def f2(x):
    print(x)
    eval = run2(test_size=x[0], neurons=x[1])
    print("Loss:{0}     Accuracy:{1}".format(eval[0], eval[1]))

    return 1 - eval[1]


""""
# hyper parameter space and minimization
spaces1 = [(180, 500), (8, 64), (32, 256)]  # test_size 10%-30%, filter 8-64, nodes 8-256
res1 = gp_minimize(f1, spaces1, n_calls=10, acq_func='EI')


# visuallising
print(res1)
plot_objective(res1)
plot_convergence(res1)


spaces2 = [(180, 540), (32, 256)]
res2 = gp_minimize(f2, spaces2, n_calls=100, acq_func='EI')
print(res2)


# save result
dump(res1, "result_cnn_pool_ei.pkl")
#dump(res2, "result_nn.pkl")
"""