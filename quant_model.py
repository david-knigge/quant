import pandas as pd
from sklearn import linear_model, model_selection
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tflearn as tf
import tensorflow

class QuantModel:

    def __init__(self, modeltype = "linreg"):
        if modeltype == "linreg":
            self.type = "LinearRegression"
        elif modeltype == "neurnet":
            self.type = "NeuralNetwork"
            net = tf.input_data([None,4])
            print(tensorflow.shape(net))
            net = tf.embedding(net, input_dim=1424, output_dim=128)
            net = tf.lstm(net, 128, dropout=0.8)
            net = tf.fully_connected(net, 2, activation='softmax')
            net = tf.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

            self.model = tf.DNN(net, tensorboard_verbose=0)


    def neural_net_train(self, input_values, expected_values):
        dates = input_values['Date'][50:]
        input_values = input_values.reindex(columns=['Close', 'macd', 'macds', 'macdh'])[50:]
        expected_values = expected_values['Change 24h'].values.reshape(1474,1)[50:]

        X_train, X_test, y_train, y_test = model_selection.train_test_split(input_values, expected_values, test_size=0.33)
        print(X_train[:10])
        print("")
        print(X_test[:10])
        print("")
        print(y_train[:10])
        print("")
        print(y_test[:10])
        print("")

        self.model.fit(X_train, y_train, show_metric=True)

    def linear_regression_model(input_values, expected_values):
        # linear regression model saven in body_regression
        body_regression = linear_model.LinearRegression()
        dates = input_values['Date']
        input_values = input_values.reindex(columns=['Price % 24h', 'Volume % 24h', 'macdh', 'rsi_14', 'gtrends'])
        body_regression.fit(input_values[50:], np.array(expected_values['Change 24h'].values.reshape(1474,1)[50:]))

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
