import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import csv


class QuantModel:

    def __init__(self):
        if modeltype == "linreg":
            self.model = "LinearRegression"
        elif modeltype == "neurnet":
            self.model = "NeuralNetwork"
            random.seed(1)

            self.synaptic_weights = 2 * random.random((features, 1)) - 1

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
        print(input_values.size)
        print(input_values['Open'].reshape(-1,1).shape)
        print(expected_values.size)
        print(expected_values['Target'].shape)
        print(len(expected_values['Target'].values))
        body_regression = linear_model.LinearRegression()
        print(input_values)
        body_regression.fit(input_values.loc['Open','Close','Volume'].as_matrix(), np.asarray(expected_values['Target'].as_matrix()).reshape(-1,1))


        plt.scatter(input_values, expected_values)
        plt.plot(input_values, body_reg.predict(input_values))
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
