import sickle

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


CUR_DIR, ROOT = sickle.paths(__file__)
VARS = ['n_UMI', 'n_genes', 'percent_mito', 'percent_ribo']


def spearman(x, y):
    "Spearman Rho"
    return scipy.stats.spearmanr(x, y)


def pearson(x, y):
    "Pearson"
    return scipy.stats.pearsonr(x, y)


def plot(x, y, xlabel=None, ylabel=None, file=None):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(
        x,
        y,
        'o',
        ms=5,
        alpha=0.4,
        markeredgewidth=0
    )
    # todo
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tick_params(labelsize=8)
    fig.subplots_adjust(0.14, 0.12, 0.98, 0.98)
    fig.savefig(
        file,
        dpi=200,
    )

def main(exp, assignment, function, plot_largest=0):
    sc01 = sickle.load_sc_scanpy(exp, assignment)

    sc01p = sickle.load_predictions(os.path.join(
        CUR_DIR,
        '{}-preds.csv'.format(exp.lower())
    ), 'MCA')

    prediction = sc01p.idxmax(axis=1)
    pred_score = sc01p.max(axis=1)

    data = sc01.obs.copy()
    data['prediction'] = prediction[data.index]
    data['pred_score'] = pred_score[data.index]

    result = []
    for x in data.prediction.unique():
        datax = data.loc[data.prediction == x, VARS + ['pred_score']]
        if datax.shape[0] < 30:
            continue
        for idx, var in enumerate(VARS):
            corr, pval = function(datax['pred_score'], datax[var])
            result.append((x, datax.shape[0], var , corr, pval))

    datax = data[VARS + ['pred_score']]
    for idx, var in enumerate(VARS):
        corr, pval = function(datax['pred_score'], datax[var])
        result.append(('All', datax.shape[0], var, corr, pval))

    result = pd.DataFrame(
        result,
        columns=['Cell type', 'Size', 'Variable', function.__doc__, 'p-value']
    )
    result = result[result['p-value'] < .05]
    above_mean = result.loc[
        result[function.__doc__].abs() > result[function.__doc__].abs().mean(),
        :
    ]
    order = above_mean[function.__doc__].abs().sort_values(ascending=False)
    above_mean.reindex(order.index).to_csv(os.path.join(
        CUR_DIR,
        'sc01-pred-score-{}.csv'.format(function.__name__)
    ), float_format='%.4g')

    out = """{}
===============
Rows: {}
Avg correlation: {:.4f}
Avg log10 pval: {}
Max corr row:
        {}
"""
    print(out.format(
        function.__doc__,
        result.shape[0],
        result[function.__doc__].abs().mean(),
        np.mean(np.log10(result.loc[result['p-value'] > 0, 'p-value'])),
        '\n\t'.join(str(result.loc[result[function.__doc__].abs().idxmax(), :]).split('\n'))
    ))

    human_names = {
        'percent_mito': 'Percentage of mitochondial genes',
        'n_UMI': 'Number of UMIs'
    }

    if plot_largest:
        for i, (cell_type, _, var, coeff, _) in enumerate(above_mean.iloc[
                :plot_largest,
                :].itertuples(index=False)):
            datax = data.loc[data.prediction == cell_type, [var, 'pred_score']]
            plot(
                datax[var],
                datax['pred_score'],
                xlabel=human_names.get(var, var),
                ylabel='Prediction probability',
                file=os.path.join(CUR_DIR, 'correlation{}.png'.format(i + 1)),
            )
            plot(
                datax[var].rank(),
                datax['pred_score'].rank(),
                xlabel=human_names.get(var, var),
                ylabel='Prediction probability',
                file=os.path.join(CUR_DIR, 'correlation{}-rank.png'.format(i + 1)),
            )



if __name__ == '__main__':
    main('SC01', 'SC01v2', spearman, 2)
    #main('SC01', 'SC01v2', pearson)
