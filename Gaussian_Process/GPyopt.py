from sklearn import datasets
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout
from keras.layers import Flatten,Dense
from keras.callbacks import EarlyStopping
import GPy,GPyOpt

class CNN:
    def __init__(self, first_input, last_output,
                 l1_out, l2_out, l1_drop, l2_drop,
                 batch_size, epochs, validation_split):
        self.first_input = first_input
        self.last_input = last_output
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.datasets()
        self.__model = self.Model()

    def datasets(self):
        digits = datasets.load_digits()
        data = digits.data
        target = digits.target
        x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2,
                                                            random_state=21)
        #normalizing to calculate faster
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 16
        x_test /= 16
        #convert vector to binary class
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
        return x_train, x_test, y_train, y_test

    def Model(self):
        model = Sequential()
        model.add(Dense(self.l1_out, activation='relu', input_shape=self.first_input))
        model.add(Dropout(self.l1_drop))
        model.add(Dense(self.l2_out, activation='softmax'))
        model.add(Dense(self.l2_drop))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                      metrics=['accuracy'])

        return model

    def fit(self):
        early_stop = EarlyStopping(patience=0, verbose=1)
        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=0,
                         validation_split=self.validation_split,
                         callbacks=[early_stop])

    def evaluate(self):
        self.fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test,
                                           batch_size=self.batch_size,
                                           verbose=0)
        return evaluation

#function to run
def run(first_input=64, last_output=10,
              l1_out=32, l2_out=32,
              l1_drop=0.2, l2_drop=0.2,
              batch_size=100, epochs=5, validation_split=0.1):
    _cnn = CNN(first_input=first_input, last_output=last_output,
                   l1_out=l1_out, l2_out=l2_out,
                   l1_drop=l1_drop, l2_drop=l2_drop,
                   batch_size=batch_size, epochs=epochs,
                   validation_split=validation_split)
    _cnn.evaluation = _CNN.evaluate()
    return _cnn.evaluation


bounds = [{'name': 'validation_split', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l1_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l2_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l1_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1)},
          {'name': 'l2_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (10, 100, 500)},
          {'name': 'epochs', 'type': 'discrete', 'domain': (5, 10, 20)}]



# function to optimize mnist model
def f(x):
    print(x)
    evaluation = run(
        l1_drop=float(x[:, 1]),
        l2_drop=float(x[:, 2]),
        l1_out=int(x[:, 3]),
        l2_out=int(x[:, 4]),
        batch_size=int(x[:, 5]),
        epochs=int(x[:, 6]),
        validation_split=float(x[:, 0]))
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]



opt_cnn = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt_cnn.run_optimization(max_iter=10)
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
\t{10}:\t{11}
\t{12}:\t{13}
""".format(bounds[0]["name"],opt_mnist.x_opt[0],
           bounds[1]["name"],opt_mnist.x_opt[1],
           bounds[2]["name"],opt_mnist.x_opt[2],
           bounds[3]["name"],opt_mnist.x_opt[3],
           bounds[4]["name"],opt_mnist.x_opt[4],
           bounds[5]["name"],opt_mnist.x_opt[5],
           bounds[6]["name"],opt_mnist.x_opt[6]))
print("optimized loss: {0}".format(opt_mnist.fx_opt))
opt_mnist.x_opt