import os

import predict
import split
import train
import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def main():
    X, y = utils.load_10x(os.path.join(ROOT, 'SC02'), 'SC02v2')
    splits = split.split(X, y)
    models = train.models(splits)

    sc03x, sc03y = utils.load_10x(os.path.join(ROOT, 'SC03'), 'SC03')
    preds = predict.predict(models, X.columns, sc03x)
    preds.to_csv(os.path.join(CUR_DIR, 'sc03-preds.csv'))


if __name__ == '__main__':
    main()
