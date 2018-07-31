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


def main():
    sc01 = utils.load_10x_scanpy(os.path.join(ROOT, 'SC01'), 'SC01v2')
    sc01p = load_predictions(os.path.join(CUR_DIR, 'sc01-best-preds.csv'))

    prediction = sc01p.idxmax(axis=1)
    pred_score = sc01p.max(axis=1)

    data = sc01.obs.copy()
    data['prediction'] = prediction[data.index]
    data['pred_score'] = pred_score[data.index]

    result = []
    for x in data.prediction.unique():
        datax = data.loc[data.prediction == x, VARS + ['pred_score']]
        corr, pval = scipy.stats.spearmanr(datax)
        for idx, var in enumerate(VARS):
            if np.isscalar(corr) and np.isnan(corr):
                rho = np.nan
                p = np.nan
            else:
                rho = corr[-1, idx]
                p = pval[-1, idx]
            result.append((x, var, datax.shape[0], rho, p))

    datax = data[VARS + ['pred_score']]
    corr, pval = scipy.stats.spearmanr(datax)
    for idx, var in enumerate(VARS):
        result.append(('All', var, datax.shape[0], corr[-1, idx], pval[-1, idx]))

    result = pd.DataFrame(
        result,
        columns=['Cell type', 'Variable', 'Size', 'Spearman Rho', 'p-value']
    )
    result[result['p-value'] < .05].to_csv(os.path.join(CUR_DIR, 'sc01-pred-score-corr.csv'))



if __name__ == '__main__':
    main()
