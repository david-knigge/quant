from quant_dataset_bitcoin import QuantDatasetBitcoin
from quant_model import QuantModel

class Quant:

    def __init__(self, args):
        self.QuantDataset = QuantDatasetBitcoin(
            dataset_path=args.get('dataset'),
            currency=args.get('currency'),
            override=args.get('override'),
            twitter=args.get('twitter')
        )

        if args.get('plot'):

            self.QuantDataset.plot()

        model = QuantModel(self.QuantDataset.dataset, self.QuantDataset.target, modeltype='neurnet', twitter=args.get('twitter'))

        X_test, y_test = model.X_test, model.y_test

        # neural_net = QuantModel.Linear_regression_model(self.getdataset(), self.gettarget())
        # model = QuantModel("neurnet")
        # model.neural_net_train(self.getdataset(), self.gettarget())

    def sim_trade(self, dataset, investment):
        pass

    def opt_trade(self, target, investment):
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
    args = p.parse_args()
    q = Quant(vars(args))
