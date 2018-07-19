import os

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def compute_score(cv, y_test):
    preds = pd.read_csv(os.path.join(CUR_DIR, 'cv{}-predictions.csv'.format(cv)), index_col=0)
    preds.columns = preds.columns.astype(int)
    final_cluster = preds.idxmax(axis=1)
    f1 = sklearn.metrics.f1_score(y_test, final_cluster, average='weighted')

    wrong_idx = y_test.index[final_cluster != y_test]
    for index in wrong_idx:
        main_cluster = preds.loc[index, :].idxmax()
        score = preds.loc[index, main_cluster]
        second_cell_type = preds.loc[index, :][preds.loc[index, :] < score].idxmax()
        final_cluster[index] = second_cell_type
    second_f1 = sklearn.metrics.f1_score(y_test, final_cluster, average='weighted')
    return f1, second_f1


def main():
    X, y = utils.load_mca_lung()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    scores = []
    for i in range(1, 6):
        for tr_idx, test_idx in splits[i - 1:i]:
            _, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
            scores.append(compute_score(i, y_test))
    scores = pd.DataFrame(scores, columns=['f1', 'second_f1'])
    scores.to_csv(os.path.join(CUR_DIR, 'scores.csv'))


if __name__ == '__main__':
    main()

