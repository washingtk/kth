from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping
from skopt import gp_minimize, dump, load
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# load data and split it into train and test set
digits = datasets.load_digits()
x = np.asarray(digits.data, "float32")
x /= 16
y = np_utils.to_categorical(digits.target, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
x_train = x_train.reshape(x_train.shape[0], 8, 8, 1)
x_test = x_test.reshape(x_test.shape[0], 8, 8, 1)


def cnn(l1, l2, l3):
    model = Sequential()
    model.add(Conv2D(l1, input_shape=(8, 8, 1), kernel_size=(3, 3), activation='relu',
                     padding='same', strides=(1, 1)))
    model.add(Conv2D(l2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(l3, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=cnn, verbose=0)


l1 = (16, 64)
l2 = (16, 64)
l3 = (32, 256)
param_grid = dict(l1=l1, l2=l2, l3=l3)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid)
grid_result = grid.result(x_test, y_test)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))