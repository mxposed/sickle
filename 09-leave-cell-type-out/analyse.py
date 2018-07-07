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
import scipy.stats.mstats

import os.path


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def load_predictions(path):
    clusters = pd.read_csv('{}/SC02v2_clusters.csv'.format(
        os.path.join(CUR_DIR, '..', '00-metadata'),
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(int)

    preds = pd.read_csv(path, index_col=0)
    preds = preds.reindex(columns=sorted(preds.columns, key=lambda x: float(x) < 0))
    new_columns = list(clusters.iloc[:,0])
    unseen_number = len(preds.columns) - len(clusters)
    if unseen_number:
        unseen_columns = list(preds.columns)[-unseen_number:]
        unseen = preds.loc[:,unseen_columns].min(axis=1)
        preds.drop(columns=unseen_columns, inplace=True)
        preds['Unseen'] = unseen
        new_columns += ['Unseen']
        #new_columns += unseen_columns
    else:
        unseen = 1 - preds.max(axis=1)
        preds['Unseen'] = unseen
        new_columns += ['Unseen']
    preds.columns = new_columns
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
    second_choices = second_choices.reindex(columns=sorted(second_choices.columns, key=lambda x: -total[x]))
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


def calc_spe_sens(truth, preds):
    PLASMA = 7
    predicted = preds.idxmax(axis=1)
    unseen_idx = set(predicted.index[predicted == 'Unseen'])
    plasma_idx = set(truth.index[truth.cluster == PLASMA])
    not_plasma_idx = set(truth.index[truth.cluster != PLASMA])
    sensitivity = len(plasma_idx & unseen_idx) / len(plasma_idx) * 100
    specificity = len(not_plasma_idx - unseen_idx) / len(not_plasma_idx) * 100
    precision = len(plasma_idx & unseen_idx) / len(unseen_idx) * 100
    return specificity, sensitivity, precision


def process(exp):
    sc03_clusters = pd.read_csv('{}/SC03_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
    ), index_col=0)
    sc03_clusters.columns = ['cluster']
    preds = load_predictions(os.path.join(CUR_DIR, '{}.csv'.format(exp)))

    # second_maxes = get_second_maxes(preds)
    # maxes_heatmap = plot_second_maxes(second_maxes)
    # maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-second-max-heatmap.png'.format(exp)))
    #
    # plasma_second_maxes = get_second_maxes(preds.loc[sc03_clusters.index[sc03_clusters.cluster == 7],:])
    # maxes_heatmap = plot_second_maxes(plasma_second_maxes)
    # plt.suptitle('Second max predictions for Plasma cells')
    # maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-plasma-second-max-heatmap.png'.format(exp)))

    hmap = heatmap(preds)
    hmap.suptitle('Predictions for SC03 dataset')
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-heatmap.png'.format(exp)))

    hmap = heatmap(preds.loc[sc03_clusters.index[sc03_clusters.cluster == 7],:], figsize=(16, 12))
    hmap.suptitle('Predictions for SC03 Plasma cells')
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-plasma-heatmap.png'.format(exp)))

    spe_sens = calc_spe_sens(sc03_clusters, preds)
    open(os.path.join(CUR_DIR, '{}-spe-sens.txt'.format(exp)), 'w').write(
        'Specificity: {:.1f}%\tSensitivity: {:.1f}%\tPrecision: {:.1f}%'.format(*spe_sens)
    )


def main():
    process('sc03-it30-oth1')
    process('sc03-it50-oth4')
    process('sc03-it100-oth2')
    process('sc03-it200-cum2')
    process('sc03-it200-cum4')
    process('sc03-it200-int2')
    process('sc03-it200-int4')


if __name__ == '__main__':
    main()
