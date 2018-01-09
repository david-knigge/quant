import sys, requests, json, datetime, os
import numpy as np
import requests
import os.path

class QuantDatasetBitcoin:

    def __init__(self, currency = "BTC", dataset = None):
        self.dataset = self.getdataset(dataset)

    def pulldata(self, currency):
        training_data = []
        r = requests.get('https://min-api.cryptocompare.com/data/pricehistorical?tsyms=USD&ts=' + str(timestamp) + '&fsym=' + currency)
        i = 0
        previous = r.json()[currency]['USD']
        while(r.status_code == requests.codes.ok and i < 500):
            clear()
            print("Retrieving training data for " + currency + " on "+ time.ctime(timestamp))
            percentage = (r.json()[currency]['USD'] - previous) / previous
            training_data.append([timestamp,i,percentage])
            i = i + 1
            timestamp -= day
            r = requests.get('https://min-api.cryptocompare.com/data/pricehistorical?tsyms=USD&ts=' + str(timestamp) + '&fsym=' + currency)
        training_data.reverse()

        for unit in training_data:
            unit[1] = len(training_data) - unit[1]
        return training_data

    def getdataset(self, currency, dataset):
        if(os.path.isfile(dataset)):
            return self.fromcsv(dataset)
        else:
            return self.tocsv(self.pulldata(currency), currency)

    def fromcsv(self, dataset):
        return np.loadtxt(dataset, delimiter=",")

    def tocsv(self, dataset, currency):
        os.path.dirname(os.path.abspath(__file__)) + "../datasets/" + currency + ".csv"
        pass
