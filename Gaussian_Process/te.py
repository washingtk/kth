from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.layers import Flatten, Dense
from skopt import gp_minimize
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load data and split it into train and test set
digits = datasets.load_digits()
x = np.asarray(digits.data, "float32")
x /= 16
y = np_utils.to_categorical(digits.target, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
x_train = x_train.reshape(x_train.shape[0], 8, 8, 1)
x_test = x_test.reshape(x_test.shape[0], 8, 8, 1)


def cnn(x):
    model = Sequential()
    model.add(Conv2D(x[0], input_shape=(8,8,1), kernel_size=(3,3), activation='relu',
                     padding='same', strides=(1,1)))
    model.add(Conv2D(x[1], kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(x[2], activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#define function of the parameters which we will minimize

def f(x):
    model = cnn(x)
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("test loss", score[0])
    print("test accuracy", score[1])

    return 1 - score[1]


#put it in minimizer function

spaces = [(8, 16, 32, 64, 128), (8, 16, 32, 64, 128), (10, 20, 30)]
res = gp_minimize(f, spaces, acq_func='EI', n_calls=10)
print(res)


""""
model = Sequential()
model.add(Conv2D(32, input_shape=(8,8,1), kernel_size=(3,3), activation='relu',
                     padding='same', strides=(1,1)))
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""