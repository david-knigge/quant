from quant_dataset_bitcoin import QuantDatasetBitcoin
from quant_model import QuantModel
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt


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
        'kullback_leibler_divergence', 'poisson']

        optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta',
        'Adam', 'Adamax', 'Nadam']

        # best_precision = 0
        # best_loss = "beep"
        # best_opt = "boop"
        # best_batch = 0
        #
        # best_list = []

        if args.get('evaluate'):
            with open("results.txt", "a") as myfile:
                #for i in range(1,5):
                    #for loss in losses:
                attempts = 5
                loss = losses[9]
                i = 3
                average = [0,0,0,0,0,0,0]
                current_stats = [' ',' ',' ',' ',' ',' ',' ']

                for j in range(attempts):
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[0])
                    average[0] += model.correct
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[1])
                    average[1] += model.correct
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[2])
                    average[2] += model.correct
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[3])
                    average[3] += model.correct
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[4])
                    average[4] += model.correct
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[5])
                    average[5] += model.correct
                    model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=i, loss_type=loss, opt=optimizers[6])
                    average[6] += model.correct

                for elem in range(7):

                    current_stats[elem] = ("Loss type: " + loss + ", optimizer: "
                    + optimizers[elem]+ ", precision: " + str(average[elem]/attempts) +
                    "%, batch size: " + str(i))

                    print("Loss type: " + loss + ", optimizer: " +
                    optimizers[elem] + ", precision: " + str(average[elem]/attempts) +
                    "%, batch size: " + str(i))

                print("Time for this loss function was: " + str(datetime.now()-latest_time))
                myfile.write(" \n \t --- time it took for this loss function: " + str(datetime.now()-latest_time) + "\n"+'\n'.join(current_stats) + "\n\n")
                latest_time = datetime.now()
                sys.stdout.flush()

                # myfile.write("\n --- Everything for batch size \"" + str(i) +
                # "\" took " + str(datetime.now() - start_time) + "seconds --- \n")

                # best_list.append("The best settings at batch size \"" + str(i) + "\" are: "+
                # "\n \t Loss type: " + best_loss +
                # "\n \t Optimizer: " + best_opt +
                # "\n \t Precision: " + str(best_precision) +
                # "\n \t Batch size: " + str(best_batch) + "\n\n")
                # sys.stdout.flush()
            # print(best_list)
                # for elem in best_list:
                #     myfile.write(str(elem))

        #X_test, y_test = model.X_test, model.y_test

        # neural_net = QuantModel.Linear_regression_model(self.getdataset(), self.gettarget())
        # model = QuantModel("neurnet")
        # model.neural_net_train(self.getdataset(), self.gettarget())l

    def plot(self):

        plt.figure(1)
        plt.subplot(2,2,1)
        QM = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h'])
        plt.plot(QM.dates, QM.model.predict(QM.input_values))

        # plt.subplot(2,2,2)
        # model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'macdh', 'rsi_14'])
        # plt.plot(QM.dates, QM.model.predict(QM.input_values))
        #
        # plt.subplot(2,2,3)
        # model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'macdh', 'gtrends'])
        # plt.plot(QM.dates, QM.model.predict(QM.input_values))
        #
        # plt.subplot(2,2,4)
        # model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=self.twitter, batches=1, loss_type='mse', opt='Nadam', variables= ['Price % 24h', 'Volume % 24h', 'rsi_14', 'gtrends'])
        # plt.plot(QM.dates, QM.model.predict(QM.input_values))

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
