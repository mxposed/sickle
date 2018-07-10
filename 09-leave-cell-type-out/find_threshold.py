import os

import numpy as np
import pandas as pd
import sklearn.metrics

import analyse
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
    clusters = pd.read_csv('{}/SC02v2_clusters.csv'.format(
        os.path.join(CUR_DIR, '..', '00-metadata'),
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(float)
    clusters.columns = ['label']

    X, y = get_reference()
    result = []
    for threshold in np.linspace(0.01, 1, 41):
        scores = []
        i = 1
        for cls in sorted(y.unique()):
            result_path = os.path.join(CUR_DIR, 'cv-{}.csv'.format(i))
            preds = analyse.load_predictions(result_path)
            predicted_cluster = pd.Series(index=preds.index)
            predicted_cluster[preds.index[preds.Unseen > threshold]] = 'Unseen'
            main_columns = list(set(preds.columns) - set(['Unseen']))
            predicted_cluster[preds.index[preds.Unseen <= threshold]] = preds.loc[preds.Unseen <= threshold, main_columns].idxmax(axis=1)

            true_cluster = y.copy()
            true_cluster = true_cluster[preds.index]
            true_cluster[true_cluster == cls] = 'Unseen'
            true_cluster = true_cluster.replace(
                {idx: val for idx, val in clusters.label.items()}
            )
            scores.append(sklearn.metrics.f1_score(true_cluster, predicted_cluster, average='weighted'))

            i += 1
        result.append(pd.DataFrame([[threshold, np.mean(scores)]]))
    result = pd.concat(result)
    result.columns = ['threshold', 'f1_score']
    result.to_csv(os.path.join(CUR_DIR, 'thresholds.csv'))


if __name__ == '__main__':
    experiment(catboost_iters=50)
