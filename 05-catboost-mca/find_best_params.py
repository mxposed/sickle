import os
import sys
import timeit

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


def save_record(record, label):
    pd.DataFrame(record, columns=['params', 'score', 'time']).to_csv(
        os.path.join(CUR_DIR, '{}.csv'.format(label))
    )


def eval_params(params, X_train, y_train, X_valid, y_valid):
    clf = catboost.CatBoostClassifier(
        l2_leaf_reg=params['l2_leaf_reg'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        iterations=params.get('iters', 50),
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='TotalF1',
        thread_count=10,
    )

    clf.fit(
        X_train,
        y_train,
    )

    y_pred = clf.predict(X_valid)
    return sklearn.metrics.f1_score(y_valid, y_pred, average='weighted')


def cross_val(X, y, params, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = []
    splits = list(skf.split(X, y))
    for i, (tr_ind, val_ind) in enumerate(splits):
        print('({}/{})'.format(i + 1, len(splits)), file=sys.stderr, end='')
        sys.stderr.flush()

        X_train, y_train = X.iloc[tr_ind, :], y.iloc[tr_ind]
        X_valid, y_valid = X.iloc[val_ind, :], y.iloc[val_ind]

        scores.append(eval_params(params, X_train, y_train, X_valid, y_valid))
        print('\b\b\b\b\b', file=sys.stderr, end='')
        sys.stderr.flush()
    return np.mean(scores)


def catboost_GridSearchCV(X, y, params_space, cv=5, splits=1, current_split=1, report):
    max_score = 0
    best_params = None
    all_params = list(ParameterGrid(params_space))
    divider = current_split - 1
    my_params = [all_params[i] for i in range(len(all_params)) if i % splits == divider]
    for i, params in enumerate(my_params):
        print('{:3d}% '.format(i * 100 // len(my_params)), file=sys.stderr, end='')
        sys.stderr.flush()
        start = timeit.default_timer()
        score = cross_val(X, y, params, cv=cv)
        if score > max_score:
            max_score = score
            best_params = params
        report.append([repr(params), score, timeit.default_timer() - start])
        print('\r' + ' ' * 20 + '\r', file=sys.stderr, end='')
        sys.stderr.flush()
    return max_score, best_params


def main(outer_split, inner_split, inner_splits, report):
    X, y = utils.load_mca_lung()
    params_space = {
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'learning_rate': np.linspace(1e-3, 8e-1, num=10),
        'depth': [6, 8, 10],
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    for tr_idx, val_idx in splits[current_split - 1:current_split]:
        print('Running {}/5 outer CV'.format(current_split))
        X_train, y_train = X.iloc[tr_idx, :], y.iloc[tr_idx]
        X_valid, y_valid = X.iloc[val_idx, :], y.iloc[val_idx]

        score, best_params = catboost_GridSearchCV(X_train, y_train,
                                                   params_space, cv=5,
                                                   splits=inner_splits,
                                                   current_split=inner_split,
                                                   report)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: {} <N> <K> <T>'.format(__file__), file=sys.stderr)
        sys.exit(1)
    outer_split = int(sys.argv[1])
    inner_split = int(sys.argv[2])
    inner_splits = int(sys.argv[3])
    record = []
    main(outer_split, inner_split, inner_splits, record)
    save_record(record,
                'search-{}+{}-of-{}'.format(outer_split, inner_split, inner_splits))
