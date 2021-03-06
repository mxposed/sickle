import pandas as pd
import numpy as np
import sklearn
import catboost
import scipy.sparse
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import os.path

import sankey


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def load_predictions(path):
    clusters = pd.read_csv(os.path.join(CUR_DIR, 'MCA_clusters.csv'))
    preds = pd.read_csv(path, index_col=0)
    preds.columns = clusters.Annotation
    preds.index = preds.index.str.replace('-1', '')
    return preds


def heatmap(predictions):
    preds = predictions.copy()
    preds.columns = list(range(len(preds.columns)))
    preds['cluster'] = preds.idxmax(axis=1)
    preds['max_score'] = predictions.max(axis=1)
    preds.sort_values(['cluster', 'max_score'], ascending=[True, False], inplace=True)
    preds = predictions.reindex(preds.index)
    plt.figure(figsize=(16, 40))
    ax = seaborn.heatmap(preds, yticklabels=[])
    return ax.get_figure()


def get_second_maxes(predictions):
    second_choices = pd.DataFrame(index=predictions.columns, columns=predictions.columns)
    second_choices.fillna(0, inplace=True)
    for cell in predictions.index:
        cell_type = predictions.loc[cell, :].idxmax()
        score = predictions.loc[cell, cell_type]
        second_cell_type = predictions.loc[cell, :][predictions.loc[cell, :] < score].idxmax()
        second_choices.loc[cell_type, second_cell_type] += 1
    sums = second_choices.sum(axis=1)
    sums[sums == 0] = 1
    second_choices = second_choices.div(sums, axis='index')
    total = second_choices.sum(axis=0)
    second_choices = second_choices.reindex_axis(sorted(second_choices.columns, key=lambda x: -total[x]), axis=1)
    return second_choices


def plot_second_maxes(maxes):
    plt.figure(figsize=(14,14))
    sums = maxes.sum(axis=0)
    ax = seaborn.clustermap(maxes,
                            xticklabels=[maxes.columns[i] + ' ({:.2f})'.format(sums[i]) for i in range(len(sums))],
                            col_cluster=False,
                           )
    ax.ax_heatmap.set_ylabel('Main cell type')
    ax.ax_heatmap.set_xlabel('Second cell type')
    return ax


def process(exp, query):
    sankey_order = pd.read_csv(
        os.path.join(os.path.dirname(CUR_DIR), 'sankey_order.csv')
    )
    exp_clusters = pd.read_csv('{}/{}_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
        query
    ), index_col=0)
    exp_clusters.columns = ['cluster']

    clusters = pd.read_csv('{}/{}_clusters.csv'.format(
        os.path.join(os.path.dirname(CUR_DIR), '00-metadata'),
        query
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(int)

    preds = '{}-best-preds.csv'.format(exp)
    sc01 = load_predictions(os.path.join(CUR_DIR, preds))
    sc01_second_maxes = get_second_maxes(sc01)
    sc01_second_maxes.sum(axis=0).to_csv(os.path.join(CUR_DIR, '{}-second-max-total.csv'.format(exp)))
    maxes_heatmap = plot_second_maxes(sc01_second_maxes)
    maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-second-max-heatmap.png'.format(exp)))

    s = sankey.sankey(
        clusters.iloc[:, 0].loc[exp_clusters.cluster],
        sc01.idxmax(axis=1)[exp_clusters.index],
        alpha=.5,
        left_order=sankey_order.order
    )
    s.savefig(os.path.join(CUR_DIR, '{}-sankey.png'.format(exp)))


    sc01 = heatmap(sc01)
    sc01.savefig(os.path.join(CUR_DIR, '{}-best-heatmap.png'.format(exp)))


def main():
    process('sc01', 'SC01v2')
    #process('sc01-noise', 'sc01-noise-preds.csv')
    process('sc02', 'SC02v2')
    process('sc03', 'SC03')


if __name__ == '__main__':
    main()
