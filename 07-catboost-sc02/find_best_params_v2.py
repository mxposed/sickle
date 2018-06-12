import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
import catboost
import sklearn.metrics

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def path(*args):
    return os.path.join(ROOT, *args)


def cross_val(X, y, params, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True)

    scores = []
    splits = list(skf.split(X, y))
    for i, (tr_ind, val_ind) in enumerate(splits):
        print('({}/{})'.format(i, len(splits)), file=sys.stderr, end='')
        sys.stderr.flush()

        X_train, y_train = X.iloc[tr_ind, :], y.iloc[tr_ind]
        X_valid, y_valid = X.iloc[val_ind, :], y.iloc[val_ind]

        clf = catboost.CatBoostClassifier(
            l2_leaf_reg=params['l2_leaf_reg'],
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            iterations=30,
            random_seed=42,
            logging_level='Silent',
            loss_function='MultiClass',
            eval_metric='TotalF1',
            thread_count=20,
        )

        clf.fit(
            X_train,
            y_train,
            cat_features=[X.shape[1] - 1],
        )

        y_pred = clf.predict(X_valid)
        score = sklearn.metrics.f1_score(y_valid, y_pred, average='weighted')
        scores.append(score)
        print('\b\b\b\b\b', file=sys.stderr, end='')
        sys.stderr.flush()
    return np.mean(scores)


def catboost_GridSearchCV(X, y, params_space, cv=5, splits=1, current_split=1):
    max_score = 0
    best_params = None
    all_params = list(ParameterGrid(params_space))
    divider = current_split - 1
    my_params = [all_params[i] for i in range(len(all_params)) if i % splits == divider]
    for i, params in enumerate(my_params):
        print('{:3d}% '.format(i * 100 // len(my_params)), file=sys.stderr, end='')
        sys.stderr.flush()
        score = cross_val(X, y, params, cv=cv)
        if score > max_score:
            max_score = score
            best_params = params
        print('\r' + ' ' * 20 + '\r', file=sys.stderr, end='')
        sys.stderr.flush()
    return max_score, best_params


def main(splits, current_split):
    X, y = utils.load_10x(path('SC02'), 'SC02v2')
    X_train, X_test, y_train, y_test =  train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42,
    )
    params_space = {
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'learning_rate': np.linspace(1e-3, 8e-1, num=10),
        'depth': [6, 8, 10],
    }
    score, best_params = catboost_GridSearchCV(X_train, y_train, params_space, cv=5,
                                               splits=splits, current_split=current_split)
    print(score)
    print(best_params)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <N> <K>'.format(__file__), file=sys.stderr)
        sys.exit(1)
    splits = int(sys.argv[1])
    my_split = int(sys.argv[2])
    main(splits, my_split)
