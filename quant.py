from quant_dataset_bitcoin import QuantDatasetBitcoin

class Quant:

    def __init__(self, args):
        self.ds = QuantDatasetBitcoin(
            dataset=args.get('dataset'),
            currency=args.get('currency'),
            override=args.get('override')
        )
        print(self.ds.dataset)

if __name__ == '__main__':
    import os
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--override', help='always pull fresh data', default=False)
    p.add_argument('--dataset', help='specify dataset path', default=os.path.dirname(os.path.abspath(__file__)) + "\\datasets\\BTC.csv")
    p.add_argument('--currency', help='specify training data currency', default="BTC")
    args = p.parse_args()
    q = Quant(vars(args))
