import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


class bpnn():
    def __init__(self, filename, activation, numbers, epoch, lr):
        self.filename = filename
        self.activation = activation
        self.numbers = numbers
        self.epoch = epoch
        self.lr = lr
        names = ['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)', 'depth of fusion(mm)', 'porosity(%)']
        data = pd.read_csv(self.filename, names=names)
        data = data.dropna()
        X_train = data.iloc[:, 0:4]
        y_train = data.iloc[:, 4:]
        self.std = preprocessing.StandardScaler()
        X_train_std = self.std.fit_transform(X_train)
        self.input_train = X_train_std
        self.output_train = y_train
        model = keras.models.Sequential([
            keras.layers.Dense(self.numbers, activation=self.activation, input_shape=self.input_train.shape[1:]),
            keras.layers.Dense(int(self.output_train.shape[1]), activation="relu")
        ])
        self.model = model

    def run(self, s):
        self.model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adagrad(learning_rate=self.lr))
        history = self.model.fit(self.input_train, self.output_train, epochs=self.epoch)

        def plot():
            plt.plot(history.epoch, history.history['loss'], "b-")
            plt.axis([0, self.epoch - 1, 0, 10])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.show()

        if s != 0:
            plot()

    def evaluate_train(self):
        model = self.model
        test = model.evaluate(self.input_train, self.output_train)
        return test

    def predict(self, i):
        model = self.model
        predict = model.predict(i)
        return predict

    def set_params(self, p1, p2):
        model = self.model
        model.get_layer(index=0).set_weights(p1)
        model.get_layer(index=1).set_weights(p2)

    def standard(self, p):
        p = pd.DataFrame(p, columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        return pd.DataFrame(self.std.transform(p), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
