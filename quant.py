from quant_dataset_bitcoin import QuantDatasetBitcoin
from quant_model import QuantModel
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class Quant:

    def __init__(self, args):
        self.twitter = args.get('twitter')
        self.QuantDataset = QuantDatasetBitcoin(
            dataset_path=args.get('dataset'),
            currency=args.get('currency'),
            override=args.get('override'),
            twitter=args.get('twitter')
        )

        if args.get('plot'):

            self.plot()
            #self.QuantDataset.plot()



        start_time = datetime.now()
        latest_time = datetime.now()

        losses = ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'hinge',
        'categorical_hinge', 'logcosh', 'binary_crossentropy',
        'kullback_leibler_divergence', 'poisson',]

        optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta',
        'Adam', 'Adamax', 'Nadam']

        best_precision = 0
        best_loss = "beep"
        best_opt = "boop"
        best_batch = 0

        best_list = []

        if args.get('evaluate'):
            for i in range(1,3):
                for loss in losses:
                    for optimizer in optimizers:
                        average = 0
                        # try:
                        for j in range(5):
                            model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizer)
                            average += model.correct

                        if (average/5) > best_precision:
                            best_precision = average/5
                            best_loss = loss
                            best_opt = optimizer
                            best_batch = i

                        print("Loss type: " + loss + ", optimizer: " + optimizer + ", precision: " + str(average/5) + "%, batch size: " + str(i) + ", time: " + str(datetime.now() - latest_time))
                        latest_time = datetime.now()
                        sys.stdout.flush()
                        # except:
                        #     print(str(loss) + ", " + str(optimizer) + ", " + str(i))
                        #     sys.stdout.flush()
                print("Everything for batch size \"" + str(i) + "\" took " + str(datetime.now() - start_time) + "seconds")
                best_list.append("The best settings at batch size \"" + str(i) + "\" are: "+
                "\n \t Loss type: " + best_loss +
                "\n \t Optimizer: " + best_opt +
                "\n \t Precision: " + str(best_precision) +
                "\n \t Batch size: " + str(best_batch))
                sys.stdout.flush()
            print(best_list)

        #X_test, y_test = model.X_test, model.y_test

        # neural_net = QuantModel.Linear_regression_model(self.getdataset(), self.gettarget())
        # model = QuantModel("neurnet")
        # model.neural_net_train(self.getdataset(), self.gettarget())l

    def plot(self):

        dates = []
        for date in self.QuantDataset.dataset['Date']:
            dates.append(datetime.fromtimestamp(date))
        dates = np.array(dates[50:]).reshape(1424,1)

        plt.figure(1)
        plt.subplot(2,2,1)
        QM = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h'])
        plt.plot(dates, QM.expected_values, color='red', ls='--')
        plt.plot(dates, QM.model.predict(QM.input_values))
        plt.ylabel('24h % change')
        plt.xlabel('Date')
        plt.title('Price , Volume')
        plt.draw()

        plt.subplot(2,2,2)
        QM = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'macdh % 24h', 'rsi_14 % 24h'])
        plt.plot(dates, QM.expected_values, color='red', ls='--')
        plt.plot(dates, QM.model.predict(QM.input_values))
        plt.ylabel('24h % change')
        plt.xlabel('Date')
        plt.title('Price, Volume, macdh, rsi_14')
        plt.draw()

        plt.subplot(2,2,3)
        QM = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'rsi_14 % 24h'])
        plt.plot(dates, QM.expected_values, color='red', ls='--')
        plt.plot(dates, QM.model.predict(QM.input_values))
        plt.ylabel('24h % change')
        plt.xlabel('Date')
        plt.title('Price, Volume, rsi_14')
        plt.draw()

        plt.subplot(2,2,4)
        QM = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'rsi_14 % 24h', 'macdh % 24h', 'gtrends % 24h'])
        plt.plot(dates, QM.expected_values, color='red', ls='--')
        plt.plot(dates, QM.model.predict(QM.input_values))
        plt.ylabel('24h % change')
        plt.xlabel('Date')
        plt.title('Price, Volume, rsi_14, macd, googletrends')
        plt.show()

        #model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=args.get('twitter'), batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'gtrends'])

        pass

if __name__ == '__main__':
    import os
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--override', help='force pull fresh data', action='store_true')
    p.add_argument('--dataset', help='specify dataset path', default=os.path.dirname(os.path.abspath(__file__)) + "/datasets/BTC-ind-trends.csv")
    p.add_argument('--currency', help='specify training data currency', default="BTC")
    p.add_argument('--plot', help='plot data', action="store_true")
    p.add_argument('--twitter', help='use sentiment data', action="store_true")
    p.add_argument('--evaluate', help='evaluate optimal settings for neurnet', action="store_true")
    args = p.parse_args()
    q = Quant(vars(args))
