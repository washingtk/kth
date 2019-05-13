# Problem 3

from sklearn import datasets
import tensorflow as tf
from tensorflow import keras
import numpy
import random

layers = tf.keras.layers

digits = datasets.load_digits()

n_classes = 10
n_samples = len(digits.images)

#Models: Fit for both 2D images and 1D feature vectors as training data.
#For a), we use two convolutional layers. Number of nodes are pretty much arbitrarily set. Dense is used as the output
#layer, with one node for each class. Activation is softmax.

#For all models, we use the Adam optimizer and the sparse categorical cross entropy loss function.

#2D
model_2D = keras.Sequential()

model_2D.add(layers.Conv2D(32, kernel_size=3, activation= tf.nn.relu, input_shape=(8,8,1)))
model_2D.add(layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu))
model_2D.add(layers.Flatten())
model_2D.add(layers.Dense(n_classes, activation=tf.nn.softmax))

model_2D.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#1D
model_1D = keras.Sequential()

model_1D.add(layers.Conv1D(32, kernel_size=3, activation= tf.nn.relu, input_shape=(64,1)))
model_1D.add(layers.Conv1D(32, kernel_size=3, activation=tf.nn.relu))
model_1D.add(layers.Flatten())
model_1D.add(layers.Dense(n_classes, activation=tf.nn.softmax))

model_1D.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#b)

#For b), we use two convolutional layers, with a pooling and a dropout layer after each convolutional layer
#Number of nodes are again pretty much arbitrarily set. Dense is again used as the output layer, with one node for each
#class and activation softmax.

#2D
advModel_2D = keras.Sequential()

advModel_2D.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(8,8,1)))
advModel_2D.add(layers.MaxPooling2D(pool_size=(2, 2)))
advModel_2D.add(layers.Dropout(0.25))

advModel_2D.add(layers.Flatten())
advModel_2D.add(layers.Dense(128, activation='relu'))
advModel_2D.add(layers.Dense(n_classes, activation='softmax'))

advModel_2D.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#1D
advModel_1D = keras.Sequential()

advModel_1D.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(64,1)))
advModel_1D.add(layers.MaxPooling1D(pool_size=2))
advModel_1D.add(layers.Dropout(0.25))

advModel_1D.add(layers.Flatten())
advModel_1D.add(layers.Dense(128, activation='relu'))
advModel_1D.add(layers.Dense(n_classes, activation='softmax'))

advModel_1D.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Save initial weights
w_init_1D = model_1D.get_weights();
w_init_2D = model_2D.get_weights();
w_init_adv1D = advModel_1D.get_weights();
w_init_adv2D = advModel_2D.get_weights();

num_iter = 10;
test_size = 100

#Investigate the effect of training sizes. The model is fit with 3 epochs.
train_size_list = [20, 50, 100, 500, 1000, 1500]
num_train_size = len(train_size_list)

test_acc_1D = numpy.zeros((num_iter, num_train_size))
test_acc_2D = numpy.zeros((num_iter, num_train_size))
test_acc_adv1D = numpy.zeros((num_iter, num_train_size))
test_acc_adv2D = numpy.zeros((num_iter, num_train_size))

for i in range(num_iter):
    j = 0
    for train_size in train_size_list:
        indices = random.sample(range(n_samples), train_size + test_size)
        train_ind = indices[:train_size]
        test_ind = indices[train_size:]

        train_images = digits.images[train_ind]
        train_images = train_images.reshape(train_size,8,8,1)

        train_vecs = digits.data[train_ind];
        train_vecs = train_vecs.reshape(train_size,64,1)

        train_targets = digits.target[train_ind]

        test_images = digits.images[test_ind];
        test_images = test_images.reshape(test_size, 8, 8, 1)

        test_vecs = digits.data[test_ind]
        test_vecs = test_vecs.reshape(test_size, 64, 1)

        test_targets = digits.target[test_ind]

        model_2D.fit(train_images, train_targets, epochs=3)
        model_1D.fit(train_vecs, train_targets, epochs=3)

        advModel_2D.fit(train_images, train_targets, epochs=3)
        advModel_1D.fit(train_vecs, train_targets, epochs=3)

        test_acc_1D[i,j] = model_1D.evaluate(test_vecs, test_targets)[1]
        test_acc_2D[i,j] = model_2D.evaluate(test_images, test_targets)[1]

        test_acc_adv1D[i,j] = advModel_1D.evaluate(test_vecs, test_targets)[1]
        test_acc_adv2D[i,j] = advModel_2D.evaluate(test_images, test_targets)[1]

        #Reset model weights
        model_1D.set_weights(w_init_1D)
        model_2D.set_weights(w_init_2D)
        advModel_1D.set_weights(w_init_adv1D)
        advModel_2D.set_weights(w_init_adv2D)

        j = j+1

#Calculate mean and standard deviations of accuracy
mean_acc_1D = numpy.mean(test_acc_1D, axis = 0)
mean_acc_2D = numpy.mean(test_acc_2D, axis = 0)
mean_acc_adv1D = numpy.mean(test_acc_adv1D, axis = 0)
mean_acc_adv2D = numpy.mean(test_acc_adv2D, axis = 0)

std_acc_1D = numpy.std(test_acc_1D, axis = 0)
std_acc_2D = numpy.std(test_acc_2D, axis = 0)
std_acc_adv1D = numpy.std(test_acc_adv1D, axis = 0)
std_acc_adv2D = numpy.std(test_acc_adv2D, axis = 0)

#Print mean and std of accuracy
print('Test set size 100 and training set sizes:')
print(*train_size_list, sep = ", ")
print("\n")

print('Mean test accuracy 1D:')
print(mean_acc_1D)
print("\n")

print('Mean test accuracy 2D:')
print(mean_acc_2D)
print("\n")

print('Mean test accuracy 1D advanced model:')
print(mean_acc_adv1D)
print("\n")

print('Mean test accuracy 2D advanced model:')
print(mean_acc_adv2D)
print("\n")

print('Test standard deviation 1D:')
print(std_acc_1D)
print("\n")

print('Test standard deviation 2D:')
print(std_acc_2D)
print("\n")

print('Test standard deviation 1D advanced model:')
print(std_acc_adv1D)
print("\n")

print('Test standard deviation 2D advanced model:')
print(std_acc_adv2D)
print("\n")
