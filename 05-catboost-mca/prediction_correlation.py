import os

import numpy as np
import pandas as pd
import scipy.stats

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))
VARS = ['n_UMI', 'n_genes', 'percent_mito']


def load_predictions(path):
    clusters = pd.read_csv(os.path.join(CUR_DIR, 'MCA_clusters.csv'))
    preds = pd.read_csv(path, index_col=0)
    preds.index = preds.index.str.replace('-1', '')
    preds.columns = clusters.Annotation
    return preds


def spearman(x, y):
    return scipy.stats.spearmanr(x, y)


def pearson(x, y):
    return scipy.stats.pearsonr(x, y)


def main(exp, assignment, function):
    sc01 = utils.load_10x_scanpy(os.path.join(ROOT, exp), assignment)
    sc01p = load_predictions(os.path.join(
        CUR_DIR,
        '{}-best-preds.csv'.format(exp.lower())
    ))

    prediction = sc01p.idxmax(axis=1)
    pred_score = sc01p.max(axis=1)

    data = sc01.obs.copy()
    data['prediction'] = prediction[data.index]
    data['pred_score'] = pred_score[data.index]

    result = []
    for x in data.prediction.unique():
        datax = data.loc[data.prediction == x, VARS + ['pred_score']]
        for idx, var in enumerate(VARS):
            corr, pval = function(datax['pred_score'], datax[var])
            result.append((x, var, datax.shape[0], corr, pval))

    datax = data[VARS + ['pred_score']]
    for idx, var in enumerate(VARS):
        corr, pval = function(datax['pred_score'], datax[var])
        result.append(('All', var, datax.shape[0], corr, pval))

    result = pd.DataFrame(
        result,
        columns=['Cell type', 'Variable', 'Size', 'Spearman Rho', 'p-value']
    )
    result[result['p-value'] < .05].to_csv(os.path.join(
        CUR_DIR,
        'sc01-pred-score-{}.csv'.format(function.__name__)
    ))



if __name__ == '__main__':
    main('SC01', 'SC01v2', spearman)
    main('SC01', 'SC01v2', pearson)
