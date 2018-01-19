import pandas as pd
from sklearn import linear_model, model_selection
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tensorflow

class QuantModel:

    variables = ['Price % 24h', 'Volume % 24h', 'macdh', 'rsi_14', 'gtrends']


    def __init__(self, input_values, expected_values, modeltype = "linreg", twitter=False, batches=1, loss_type="mse", opt="Nadam", variables=variables):
        if not twitter:
            if modeltype == "linreg":
                self.type = "LinearRegression"
                self.model = self.linear_regression_model(input_values, expected_values, variables)
            elif modeltype == "neurnet":
                self.type = "NeuralNetwork"
                self.model = self.neural_net(input_values, expected_values, batches, loss_type, opt, variables)
                self.correct = self.validate_sign(self.model.predict(self.X_test), self.y_test)
                self.classes = self.validate_classes(self.model.predict(self.X_test), self.y_test, [-0.3, 0.3])
        else:
            if modeltype == "linreg":
                self.type = "LinearRegression"
                self.model = self.linear_regression_model(input_values, expected_values, variables)
            elif modeltype == "neurnet":
                self.type = "NeuralNetwork"
                self.model = self.neural_net_sent(input_values, expected_values, batches, loss_type, opt, variables)
                self.correct = self.validate_sign(self.model.predict(self.X_test), self.y_test)
                self.classes = self.validate_classes(self.model.predict(self.X_test), self.y_test, [-0.3, 0.3])

    def neural_net(self, input_values, expected_values, batches, loss_type, opt, variables):
        model = Sequential()

        self.dates = input_values['Date'][50:].values
        self.input_values = input_values.reindex(columns=variables)[50:].values
        self.expected_values = expected_values['Change 24h'].values.reshape(1474,1)[50:]
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.input_values, self.expected_values, test_size=0.33)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        self.input_values = self.input_values.reshape((self.input_values.shape[0], 1, self.input_values.shape[1]))

        model.add(LSTM(
            500,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        model.add(Dense(1))
        model.compile(loss=loss_type, optimizer=opt) # Nadam is heel bueno

        history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=batches, validation_data=(self.X_test, self.y_test), verbose=2, shuffle=False)
        return model


    def linear_regression_model(input_values, expected_values, variables):
        # linear regression model saven in body_regression
        body_regression = linear_model.LinearRegression()

        dates = input_values['Date'][50:].values
        input_values = input_values.reindex(columns=variables)[50:].values
        expected_values = expected_values['Change 24h'].values.reshape(1474,1)[50:]
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(input_values, expected_values, test_size=0.33)

        body_regression.fit(input_values[50:], expected_values)
        return body_regression


    def neural_net_sent(self, input_values, expected_values, batches, loss_type, opt, variables):

        variables.append("tsentiment")
        model = Sequential()

        self.dates = input_values['Date'][50:].values
        self.input_values = input_values.reindex(columns=variables).values
        self.expected_values = expected_values['Change 24h'].values.reshape(119,1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.input_values, self.expected_values, test_size=0.33)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        input_values = input_values.reshape((input_values.shape[0], 1, input_values.shape[1]))

        model.add(LSTM(
            500,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        model.add(Dense(1))
        model.compile(loss=loss_type, optimizer=opt) # Nadam is heel bueno
        history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=batches, validation_data=(self.X_test, self.y_test), verbose=0, shuffle=False)

        return model

    def validate_sign(self, predictions, targets):
        correct = 0
        for index, value in enumerate(predictions):
            if np.sign(value) == np.sign(targets[index]):
                correct += 1
        return (correct / len(predictions)) * 100


    def validate_classes(self, predictions, targets, thresholds):
        correct = 0
        predicted = 0
        #print(min(predictions))
        #print(max(predictions))
        for index, value in enumerate(predictions):
            if value > thresholds[1] or value < thresholds[0]:
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
