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


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def load_predictions(path):
    clusters = pd.read_csv('{}/SC02_clusters.csv'.format(
        os.path.join(CUR_DIR, '..', '00-metadata'),
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(int)
    clusters.iloc[:, 0] = clusters.iloc[:, 0] + ' ' + clusters.index.astype(str)
    preds = pd.read_csv(path, index_col=0)
    preds.columns = clusters.iloc[:, 0]
    return preds


def heatmap(predictions, figsize=(16, 20)):
    preds = predictions.copy()
    preds.columns = list(range(len(preds.columns)))
    preds['cluster'] = preds.idxmax(axis=1)
    preds['max_score'] = predictions.max(axis=1)
    preds.sort_values(['cluster', 'max_score'], ascending=[True, False], inplace=True)
    preds = predictions.reindex(preds.index)
    plt.figure(figsize=figsize)
    ax = seaborn.heatmap(preds, yticklabels=[])
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


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


def process(exp):
    sc03_clusters = pd.read_csv('{}/SC03_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
    ), index_col=0)
    sc03_clusters.columns = ['cluster']
    preds = load_predictions(os.path.join(CUR_DIR, '{}-preds.csv'.format(exp)))

    second_maxes = get_second_maxes(preds)
    maxes_heatmap = plot_second_maxes(second_maxes)
    maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-second-max-heatmap.png'.format(exp)))

    plasma_second_maxes = get_second_maxes(preds.loc[sc03_clusters.index[sc03_clusters.cluster == 7],:])
    maxes_heatmap = plot_second_maxes(plasma_second_maxes)
    plt.suptitle('Second max predictions for Plasma cells')
    maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-plasma-second-max-heatmap.png'.format(exp)))

    hmap = heatmap(preds)
    hmap.suptitle('Predictions for SC03 dataset')
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-heatmap.png'.format(exp)))


    hmap = heatmap(preds.loc[sc03_clusters.index[sc03_clusters.cluster == 7],:], figsize=(16, 12))
    hmap.suptitle('Predictions for SC03 Plasma cells')
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-plasma-heatmap.png'.format(exp)))




def main():
    process('sc03')


if __name__ == '__main__':
    main()
