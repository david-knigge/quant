import pandas as pd
from sklearn import linear_model, model_selection
import numpy as np
import matplotlib.pyplot as plt
import csv
import tflearn as tf


class QuantModel:

    def __init__(self, modeltype = "linreg"):
        if modeltype == "linreg":
            self.model = "LinearRegression"
        elif modeltype == "neurnet":
            self.model = "NeuralNetwork"
            net = tf.input_data([1424,4])
            net = tf.embedding(net, input_dim=1424, output_dim=128)
            net = tf.lstm(net, 128, dropout=0.8)
            net = tf.fully_connected(net, 2, activation='softmax')
            net = tf.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

            model = tf.DNN(net, tensorboard_verbose=0)
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

    def neural_net_train(self, input_values, expected_values):
        dates = input_values['Date'][50:]
        input_values = input_values.reindex(columns=['Close', 'macd', 'macds', 'macdh'])[50:]
        expected_values = expected_values['Change 24h'].values.reshape(1474,1)[50:]

        X_train, X_test, y_train, y_test = model_selection.train_test_split(input_values, expected_values, test_size=0.33)

        model.fit(X_train, y_train, validation_set(X_test, y_test), show_metric=True)

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
        #plt.show()
        return plt
"""
if __name__ = '__main__':
    neural_net = NeuralNetwork()

    print("start weights: \n" + str(neural_net.synaptic_weights))

    neural_net.train(input_values, expected_values, 10000)

    print("new weights after training: \n"+str(neural_net.synaptic_weights))
    print("prediction for "+ str(predict_date) + " is: \n" + neural_net.predict)
"""
