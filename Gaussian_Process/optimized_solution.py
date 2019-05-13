from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping
from skopt import gp_minimize
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

# typical convolution network


def cnn(l1, l2, p1, l3, batch_size, epochs):
    model = Sequential()
    model.add(Conv2D(l1, input_shape=(8,8,1), kernel_size=(3,3), activation='relu',
                     padding='same', strides=(1,1)))
    model.add(Conv2D(l2, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(p1))
    model.add(Flatten())
    model.add(Dense(l3, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    eary_stop = EarlyStopping(patience=1, verbose=0)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[eary_stop],
              validation_split=0.1)
    return model

# function regarded as a function of hyper parameters


def f(l):
    model = cnn(l[0], l[1], l[2], l[3], l[4], l[5])
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:{0} /t Accuracy:{1}".format(score[0], score[1]))
    return 1 - score[1]

# put it in minimizer
spaces = [(16, 32, 64), (16, 32, 64), (0.1, 0.5, "uniform"), (32, 64, 128), (32, 64, 128, 256), (5, 10, 15)]
res = gp_minimize(f, spaces, n_calls=10, acq_func='EI')
print(res)