import sys, requests, json, datetime, os, time
import numpy as np
import requests
import os.path
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
from stockstats import StockDataFrame as Sdf
from quant_google_trends import QuantGoogleTrends
from collections import Counter

pd.options.mode.chained_assignment = None  # default='warn'

# create / load bitcoin dataset
class QuantDatasetBitcoin:

    # instantiate vars
    day = 86400
    now = round(time.time())
    dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/BTC-ind-trends.csv"
    target_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/targets.csv"
    indicators = ['macd','macds','macdh','rsi_14']
    thresholds = [-5, 5]

    # create dataset
    def __init__(self, currency = "BTC", dataset_path = dataset_path, override=False):
        self.override = override
        self.dataset = self.getdataset(currency, dataset_path)
        self.target = self.gettargetdata(thresholds = self.thresholds)

    #
    def gettrenddataset(self, currency):
        # add stock data
        stockdataset = self.pullstockdata(currency)
        self.augmented_stockdataset = self.augmentstockdataset(stockdataset)

        # add google trends data
        gtrends = QuantGoogleTrends()
        gtrends_dated = gtrends.gettrends([
            stockdataset['Date'].iloc[-1],
            stockdataset['Date'].iloc[0]
        ])
        gtrends_stockdata = self.augmented_stockdataset.copy(deep=True)
        gtrends_stockdata['gtrends'] = gtrends_dated.values

        self.tocsv(gtrends_stockdata, 'BTC-ind-trends')
        return gtrends_stockdata


    # retrieve data from cmc
    def pullstockdata(self, currency):
        training_data = []
        timestamp = self.now
        r = requests.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20180108')
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

        dfdata = pd.DataFrame(data[::-1], columns=headers)
        return dfdata

    # check whether a dataset is saved in given directory, else pull fresh data
    def getdataset(self, currency, dataset_path):
        if dataset_path and not self.override:
            if(os.path.isfile(dataset_path)):
                return self.fromcsv(dataset_path)
        return self.gettrenddataset(currency)

    # get data from csv file
    def fromcsv(self, dataset_path):
        return pd.read_csv(dataset_path)

    # save data to csv file
    def tocsv(self, dataset, currency):
        fname = "datasets/" + currency + ".csv"
        open(fname, 'a').close()
        dataset.to_csv(fname)

    # remove all rows that have nan values (~may 2013 - aug 2013)
    def cleandata(self, dirty):
        return dirty[~np.isnan(dirty).any(axis=1)]

    # plot the dataset
    def plot(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        dates = []
        for date in self.dataset['Date']:
            dates.append(datetime.datetime.fromtimestamp(date))

        ax1.set_xlabel('time (UNIX)')
        ax1.plot_date(dates, self.target['Target'], color='red')
        ax2.plot_date(dates, self.dataset['Volume'], '-', color='orange',)
        ax3.plot_date(dates, self.dataset['Close'], '-')
        plt.show()

    # append indicators to the dataset
    def augmentstockdataset(self, dataset):
        stockdataset = Sdf.retype(dataset.copy(deep=True))
        augmented_dataset = dataset.copy(deep=True)

        for indicator in self.indicators:
            augmented_dataset[indicator] = stockdataset[indicator].values
        return augmented_dataset

    # get target data values from file or calculate them
    def gettargetdata(self, thresholds):
        if (not self.override) and os.path.isfile(self.target_path):
            return self.fromcsv(self.target_path)
        else:
            dataset = self.dataset.copy(deep=True)
            target_values = []
            raw_target_values = []
            raw_target_values_48h = []
            for index, sample in dataset.iloc[:-2].iterrows():
                abschange = ((dataset.iloc[index + 1]['Close'] - sample['Close']) / sample['Close']) * 100
                abschange48h = (((dataset.iloc[index + 1]['Close'] + dataset.iloc[index + 2]['Close']) / 2 - sample['Close']) / sample['Close']) * 100
                change = self.hround(abschange)
                raw_target_values.append(abschange)
                raw_target_values_48h.append(abschange48h)
                if change >= thresholds[1]:
                    target_values.append(1)
                elif change <= thresholds[0]:
                    target_values.append(-1)
                else:
                    target_values.append(0)

            # to compensate for last matrix value missing
            target_values.append(0)
            raw_target_values.append(0)
            target_values.append(0)
            raw_target_values.append(0)
            target = pd.DataFrame(pd.Series(target_values), columns=['Target'])
            target['Change 24h'] = pd.Series(raw_target_values)
            target['Change 48h'] = pd.Series(raw_target_values_48h)
            self.tocsv(target, 'targets')
            return target

    def hround(self, x, base=1):
        return int(base * round(float(x)/base))
