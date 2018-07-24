import os

import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

import sankey
import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


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


def process(exp, reference, query):
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

    preds = utils.load_predictions(
        os.path.join(CUR_DIR, '{}-preds.csv'.format(exp)),
        reference
    )

    second_maxes = get_second_maxes(preds)
    maxes_heatmap = plot_second_maxes(second_maxes)
    maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-second-max-heatmap.png'.format(exp)))

    if query == 'SC03':
        plasma_second_maxes = get_second_maxes(preds.loc[exp_clusters.index[exp_clusters.cluster == 7],:])
        maxes_heatmap = plot_second_maxes(plasma_second_maxes)
        plt.suptitle('Second max predictions for Plasma cells')
        maxes_heatmap.savefig(os.path.join(CUR_DIR, '{}-plasma-second-max-heatmap.png'.format(exp)))

    hmap = heatmap(preds)
    hmap.suptitle('Predictions for {} dataset'.format(query))
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-heatmap.png'.format(exp)))

    if query == 'SC03':
        hmap = heatmap(preds.loc[exp_clusters.index[exp_clusters.cluster == 7],:], figsize=(16, 12))
        hmap.suptitle('Predictions for SC03 Plasma cells')
        hmap.subplots_adjust(top=0.88)
        hmap.savefig(os.path.join(CUR_DIR, '{}-plasma-heatmap.png'.format(exp)))

    s = sankey.sankey(clusters.iloc[:, 0].loc[exp_clusters.cluster], preds.idxmax(axis=1), alpha=.5)
    s.savefig(os.path.join(CUR_DIR, '{}-sankey.png'.format(exp)))


def main():
    process('sc02', 'SC01v2', 'SC02v2')


if __name__ == '__main__':
    main()
