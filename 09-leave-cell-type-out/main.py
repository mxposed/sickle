import os

import cross_val as cv
import predict
import split
import train
import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def get_reference():
    global _reference
    if '_reference' not in globals():
        _reference = utils.load_10x(os.path.join(ROOT, 'SC02'), 'SC02v2')
    return _reference


def experiment(**kwargs):
    X, y = get_reference()
    i = 1
    for X_train, y_train, X_test, y_test in cv.leave_cell_type_out(X, y):
        splits = split.split(
            X_train, y_train,
            other_proportion=4,
            split_order='cumsum',
        )
        models = train.models(splits, iterations=kwargs['catboost_iters'], label=label)
        result_path = os.path.join(CUR_DIR, 'cv-{}.csv'.format(i))
        predict.predict(models, X_train.columns, X_test).to_csv(result_path)
        i += 1


if __name__ == '__main__':
    experiment(catboost_iters=50)
