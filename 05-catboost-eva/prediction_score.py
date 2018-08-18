import os
import sys

import catboost
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
#import timeit

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def eval_params(params, X_train, y_train, X_valid, y_valid):
    clf = catboost.CatBoostRegressor(
        iterations=100,
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_seed=42,
        logging_level='Silent',
        thread_count=20,
        loss_function='MAE',
    )

    clf.fit(
        X_train,
        y_train,
    )

    y_pred = clf.predict(X_valid)
    return sklearn.metrics.mean_absolute_error(y_valid, y_pred)


def cross_val(X, y, params, cv=5):
    skf = sklearn.model_selection.KFold(n_splits=cv, shuffle=True, random_state=42)

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


def find_best_params(X, y, params_space, splits=1, current_split=1):
    min_score = None
    best_params = None
    all_params = list(sklearn.model_selection.ParameterGrid(params_space))
    divider = current_split - 1
    my_params = [all_params[i] for i in range(len(all_params)) if i % splits == divider]
    for i, params in enumerate(my_params):
        print('{:3d}% '.format(i * 100 // len(my_params)), file=sys.stderr, end='')
        sys.stderr.flush()
        #start = timeit.default_timer()
        score = cross_val(X, y, params)
        if min_score is None or score < min_score:
            min_score = score
            best_params = params
        #record.append([repr(params), score, timeit.default_timer() - start])
        print('\r' + ' ' * 20 + '\r', file=sys.stderr, end='')
        sys.stderr.flush()
    return min_score, best_params


def process():
    X, _ = utils.load_10x(os.path.join(ROOT, 'SC01'), 'SC01v2')
    X.drop(columns=['Batch'], inplace=True)

    predictions = pd.read_csv(os.path.join(CUR_DIR, 'sc01-best-preds.csv'), index_col=0)
    predictions.index = predictions.index.str.replace('-1', '')
    y = predictions.loc[X.index, :].max(axis=1)

    params_space = {
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'learning_rate': np.linspace(1e-3, 8e-1, num=10),
        'depth': [6, 8, 10],
    }

    score, best_params = find_best_params(X, y, params_space)
    print('Best params score: {}'.format(score))
    print('Best params: {}'.format(repr(best_params)))

    model = catboost.CatBoostRegressor(
        iterations=500,
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        random_seed=42,
        logging_level='Silent',
        thread_count=20,
        loss_function='MAE',
    )
    model.fit(X, y)
    model.save_model(os.path.join(CUR_DIR, 'sc01-pred-score.cbm'))

    importances = pd.DataFrame(model._feature_importance, X.columns)
    importances[importances[0] > 0].sort_values(
        0,
        ascending=False
    ).to_csv(os.path.join(CUR_DIR, 'sc01-pred-score-features.csv'))


def main():
    process()


if __name__ == '__main__':
    main()
