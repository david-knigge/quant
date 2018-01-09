import sys, requests, json, datetime, os, time
import numpy as np
import requests
import os.path
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
from stockstats import StockDataFrame as Sdf

# create / load bitcoin dataset
class QuantDatasetBitcoin:

    # instantiate vars
    day = 86400
    now = round(time.time())
    dataset = os.path.dirname(os.path.abspath(__file__)) + "/datasets/BTC.csv"

    # create dataset
    def __init__(self, currency = "BTC", dataset = dataset, override=False):
        self.override = override
        self.dataset = self.getdataset(currency, dataset)
        self.stockdataset = Sdf.retype(self.dataset.copy(deep=True))


    # retrieve data from cmc
    def pulldata(self, currency):
        training_data = []
        timestamp = self.now
        r = requests.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20180109')
        if r.status_code == requests.codes.ok:
            soup = BeautifulSoup(r.content, 'lxml')
            table = soup.find_all('table')[0]
            headers = [h.string for h in table.find_all('th')]
            table_body = table.find_all('tbody')[0]
            rows = table_body.find_all('tr')
            columns = [row.find_all('td') for row in rows]
            matrix = [[entry.string.replace(",","") for entry in row] for row in columns]
            for row in matrix:
                row[0] = round(time.mktime(datetime.datetime.strptime(row[0], "%b %d %Y").timetuple()))
        else:
            sys.exit("CMC Unavailable")

        # check for nan's
        for rindex, row in enumerate(matrix):
            for cindex, entry in enumerate(row):
                if entry == "-":
                    matrix[rindex][cindex] = "nan"

        # clean data and retype
        data = self.cleandata(np.array(matrix).astype(np.float))
        # save to csv
        self.tocsv(data, currency, headers)
        return pd.DataFrame(data, columns=headers)

    # check whether a dataset is saved in given directory, else pull fresh data
    def getdataset(self, currency, dataset):
        if dataset and not self.override:
            if(os.path.isfile(dataset)):
                return self.fromcsv(dataset)
        return self.pulldata(currency)

    # get data from csv file
    def fromcsv(self, dataset):
        return pd.read_csv(dataset)

    # save data to csv file
    def tocsv(self, dataset, currency, headers):
        fname = os.path.dirname(os.path.abspath(__file__)) + "/datasets/" + currency + ".csv"
        open(fname, 'a').close()
        fmt='%s, %s, %s, %s, %s, %s, %s'
        np.savetxt(fname, dataset, fmt=fmt, header=",".join(headers))
        return "asdf"

    # remove all rows that have nan values (~may 2013 - aug 2013)
    def cleandata(self, dirty):
        return dirty[~np.isnan(dirty).any(axis=1)]

    # plot the dataset
    def plot(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(self.dataset['# Date'], self.dataset['Open'])
        ax1.set_xlabel('time (UNIX)')
        ax2.plot(self.dataset['# Date'], self.dataset['Volume'], color='orange')
        plt.show()
        pass
