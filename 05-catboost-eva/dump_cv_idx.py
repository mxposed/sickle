import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def path(*args):
    return os.path.join(ROOT, *args)


def dump(*args):
    d = os.path.join(CUR_DIR, 'cv-idx')
    if not os.path.exists(d):
        os.mkdir(d)
    return os.path.join(d, *args)


def main():
    X, y = utils.load_mca_lung()
    params_space = {
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'learning_rate': np.linspace(1e-3, 8e-1, num=10),
        'depth': [6, 8, 10],
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    for i, (tr_idx, test_idx) in enumerate(splits):
        y_train = y.iloc[tr_idx]
        y_test = y.iloc[test_idx]
        
        pd.Series(y_train.index).to_csv(dump(
            'cv{}-train.csv'.format(i + 1)
        ))
        pd.Series(y_test.index).to_csv(dump(
            'cv{}-test.csv'.format(i + 1)
        ))


if __name__ == '__main__':
    main()
