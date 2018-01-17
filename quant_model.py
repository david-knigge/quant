import pandas as pd
from sklearn import linear_model, model_selection
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tflearn as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tensorflow

class QuantModel:

    def __init__(self, input_values, expected_values, modeltype = "linreg"):

        dates = input_values['Date'][50:]
        input_values = input_values.reindex(columns=['Price % 24h', 'Volume % 24h', 'macdh', 'rsi_14', 'gtrends'])[50:].values
        expected_values = expected_values['Change 24h'].values.reshape(1474,1)[50:]
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(input_values, expected_values, test_size=0.33)

        if modeltype == "linreg":
            self.type = "LinearRegression"
            self.model = self.linear_regression_model(input_values, expected_values, dates)
        elif modeltype == "neurnet":
            self.type = "NeuralNetwork"
            self.model = self.neural_net(input_values, expected_values, dates)
            print(self.validate_sign(self.model.predict(self.X_test), self.y_test))
            print(self.validate_classes(self.model.predict(self.X_test), self.y_test, [3, -3]))


            #print(tensorflow.shape(net))


            #net = tf.input_data([None,4])
            #net = tf.embedding(net, input_dim=1424, output_dim=128)
            #net = tf.lstm(net, 128, dropout=0.8)
            #net = tf.fully_connected(net, 2, activation='softmax')
            #net = tf.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

            #self.model = tf.DNN(net, tensorboard_verbose=0)


    def neural_net(self, input_values, expected_values, dates):
        model = Sequential()

        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        input_values = input_values.reshape((input_values.shape[0], 1, input_values.shape[1]))

        model.add(LSTM(
            50,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')


        history = model.fit(self.X_train, self.y_train, epochs=10, batch_size=10, validation_data=(self.X_test, self.y_test), verbose=2, shuffle=False)

        return model
        # plt.plot(dates, expected_values)
        # plt.plot(dates, model.predict(input_values))
        # plt.legend()
        # plt.show()

        # self.model.fit(X_train, y_train, show_metric=True)

    def linear_regression_model(input_values, expected_values, dates111):
        # linear regression model saven in body_regression
        body_regression = linear_model.LinearRegression()
        input_values = input_values.reindex(columns=['Price % 24h', 'Volume % 24h', 'macdh', 'rsi_14', 'gtrends'])
        expected_values = expected_values['Change 24h'].values.reshape(1474,1)[50:]

        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(input_values, expected_values, test_size=0.33)

        body_regression.fit(input_values[50:], expected_values)

        # plt.plot(dates.values.reshape(1474,1)[50:], body_regression.predict(input_values[50:]))
        # plt.show()
        return body_regression


    def validate_sign(self, predictions, targets):

        correct = 0
        for index, value in enumerate(predictions):
            if np.sign(value) == np.sign(targets[index]):
                correct += 1

        return (correct / len(predictions)) * 100


    def validate_classes(self, predictions, targets, thresholds):

        correct = 0
        predicted = 0
        for index, value in enumerate(predictions):
            if value > thresholds[1] or value < -thresholds[0]:
                if np.sign(value) == np.sign(targets[index]):
                    correct += 1
                    predicted += 1

        return (correct / predicted) * 100, predicted


"""
if __name__ = '__main__':
    neural_net = NeuralNetwork()

    print("start weights: \n" + str(neural_net.synaptic_weights))

    neural_net.train(input_values, expected_values, 10000)

    print("new weights after training: \n"+str(neural_net.synaptic_weights))
    print("prediction for "+ str(predict_date) + " is: \n" + neural_net.predict)
"""
