import os

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


def get_query():
    global _query
    if '_query' not in globals():
        _query = utils.load_10x(os.path.join(ROOT, 'SC03'), 'SC03')
    return _query


def experiment(label=None, **kwargs):
    result_path = os.path.join(CUR_DIR, 'sc03-{}.csv'.format(label))
    if not os.path.exists(result_path):
        X, y = get_reference()
        splits = split.split(X, y, other_proportion=kwargs['other_proportion'])
        models = train.models(splits, iterations=kwargs['catboost_iters'], label=label)

        sc03x, sc03y = get_query()
        predict.predict(models, X.columns, sc03x).to_csv(result_path)


if __name__ == '__main__':
    experiment(label='it30-oth1', catboost_iters=30, other_proportion=1)
    experiment(label='it100-oth2', catboost_iters=100, other_proportion=2)
