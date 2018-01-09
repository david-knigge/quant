from quant_dataset_bitcoin import QuantDatasetBitcoin

class Quant:

    def __init__(self, args):
        self.ds = QuantDatasetBitcoin(
            dataset_path=args.get('dataset'),
            currency=args.get('currency'),
            override=args.get('override')
        )

        if args.get('plot'):
            self.ds.plot()

    def getdataset(self, augmented=True):
        if augmented:
            return self.ds.augmented_dataset
        else:
            return self.ds.dataset

if __name__ == '__main__':
    import os
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--override', help='force pull fresh data', action='store_true')
    p.add_argument('--dataset', help='specify dataset path', default=os.path.dirname(os.path.abspath(__file__)) + "/datasets/BTC.csv")
    p.add_argument('--currency', help='specify training data currency', default="BTC")
    p.add_argument('--plot', help='plot data', action="store_true")
    args = p.parse_args()
    q = Quant(vars(args))
