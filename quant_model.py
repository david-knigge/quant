import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import csv
import tflearn as tf


class QuantModel:

    def __init__(self):
        if modeltype == "linreg":
            self.model = "LinearRegression"
        elif modeltype == "neurnet":
            self.model = "NeuralNetwork"
            net = tf.input_data([1714,1714])
            """random.seed(1)

            self.synaptic_weights = 2 * random.random((features, 1)) - 1
"""
    def sigmoid(self, x):
        return 1/(1+ exp(-x))

    def sigmoid_deriv(self, x):
        return x * ( 1 - x )

    def predict(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, train_input, train_output, iterations):
        for i in xrange(iterations):
            output = self.predict(train_input, )

            error = train_output - output
            weight_update = dot(train_input.T, error * self.sigmoid_deriv(output))

            self.synaptic_weights += weight_update

    def Linear_regression_model(input_values, expected_values):
        # linear regression model saven in body_regression
        body_regression = linear_model.LinearRegression()
        dates = input_values['Date']
        input_values = input_values.reindex(columns=['Close', 'macd', 'macds', 'macdh'])

        body_regression.fit(input_values[50:], np.array(expected_values['Change 24h'].values.reshape(1474,1)[50:]))


        #plt.scatter(input_values, np.asarray(expected_values['Target']).reshape(1474,1))
        # print("The values: ")
        # print(dates.values.reshape(1474,1)[50:])
        # print("The values: ")
        # print(input_values[50:])
        # print("The shape: ")
        # print(dates.values.reshape(1474,1)[50:].shape)
        # print("The shape: ")
        # print(input_values[50:].shape)
        plt.plot(dates.values.reshape(1474,1)[50:], body_regression.predict(input_values[50:]))
        plt.show()
        return plt
"""
if __name__ = '__main__':
    neural_net = NeuralNetwork()

    print("start weights: \n" + str(neural_net.synaptic_weights))

    neural_net.train(input_values, expected_values, 10000)

    print("new weights after training: \n"+str(neural_net.synaptic_weights))
    print("prediction for "+ str(predict_date) + " is: \n" + neural_net.predict)
"""
