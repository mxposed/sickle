import sickle

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn

import sankey
import utils


CUR_DIR, ROOT = sickle.paths(__file__)


def load_predictions(path):
    clusters = pd.read_csv('{}/SC02v2_clusters.csv'.format(
        os.path.join(CUR_DIR, '..', '00-metadata'),
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(int)

    preds = pd.read_csv(path, index_col=0)
    preds = preds.reindex(columns=sorted(preds.columns, key=lambda x: (float(x) < 0, float(x))))
    new_columns = list(clusters.iloc[:,0])
    unseen_number = len(preds.columns) - len(clusters)
    if unseen_number:
        unseen_columns = list(preds.columns)[-unseen_number:]
        unseen = preds.loc[:,unseen_columns].min(axis=1)
        preds.drop(columns=unseen_columns, inplace=True)
        preds['Novel cell type'] = unseen
        new_columns += ['Novel cell type']
        #new_columns += unseen_columns
    else:
        unseen = 1 - preds.max(axis=1)
        preds['Novel cell type'] = unseen
        new_columns += ['Novel cell type']
    preds.columns = new_columns
    sums = preds.sum(axis=1)
    return preds.div(sums, axis=0)


def heatmap(predictions, label=None, figsize=(20, 8), threshold=0):
    preds = predictions.copy()
    preds.columns = list(range(len(preds.columns)))
    preds['cluster'] = preds.idxmax(axis=1)
    preds['max_score'] = predictions.max(axis=1)
    preds.sort_values(['cluster', 'max_score'], ascending=[True, False], inplace=True)
    preds = predictions.reindex(preds.index)

    plt.figure(figsize=figsize)
    seaborn.set(font_scale=2.2)
    ax = seaborn.heatmap(
        preds.T,
        xticklabels=[],
        cmap="Blues",
        cbar_kws={
            'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]
        }
    )

    if label:
        ax.set_xlabel(label)
    else:
        ax.set_xlabel('')
    ax.set_ylabel('')
    ax.collections[0].set_clim(0, 1)

    ax.figure.axes[-1].tick_params(labelsize=10)
    ax.figure.axes[-1].set_ylabel('Prediction probability')

    ax.figure.tight_layout()
    ax.figure.subplots_adjust(right=1.04)
    seaborn.set(font_scale=1)
    return ax.figure


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


def calc_spe_sens(truth, preds, threshold=0):
    PLASMA = 7
    main_columns = list(set(preds.columns) - set(['Novel cell type']))
    predicted = preds.idxmax(axis=1)
    #predicted.loc[preds['Novel cell type'] > threshold] = 'Novel cell type'
    unseen_idx = set(predicted.index[predicted == 'Novel cell type'])
    plasma_idx = set(truth.index[truth == PLASMA])
    not_plasma_idx = set(truth.index[truth != PLASMA])
    sensitivity = len(plasma_idx & unseen_idx) / len(plasma_idx) * 100
    specificity = len(not_plasma_idx - unseen_idx) / len(not_plasma_idx) * 100
    precision = len(plasma_idx & unseen_idx) / len(unseen_idx) * 100
    return specificity, sensitivity, precision


def process(exp, reference, query):
    exp_clusters = sickle.assignments(query)

    clusters = sickle.load_clusters(query)

    preds = load_predictions(os.path.join(CUR_DIR, '{}.csv'.format(exp)))

    hmap = heatmap(preds)
    hmap.savefig(os.path.join(CUR_DIR, '{}-heatmap.png'.format(exp)))

    hmap = heatmap(
        preds.loc[exp_clusters.index[exp_clusters == 7], :],
        label='$\it{Plasma\ cells}$ from SC03 dataset',
        figsize=(16, 8)
    )
    hmap.savefig(os.path.join(CUR_DIR, '{}-plasma-heatmap.png'.format(exp)))

    mapping = sickle.mapping(query, reference)
    s = sankey.sankey(
        clusters.iloc[:, 0].loc[exp_clusters],
        preds.idxmax(axis=1),
        alpha=.7,
        left_order=sickle.sankey_order(),
        mapping=mapping
    )
    s.savefig(os.path.join(CUR_DIR, '{}-sankey.png'.format(exp)))

    if mapping:
        open(os.path.join(CUR_DIR, '{}-f1.txt'.format(exp)), 'w').write(
            'F1 score: {:.4f}'.format(
                mapping.f1(clusters.iloc[:, 0].loc[exp_clusters],
                           preds.idxmax(axis=1))
            )
        )

    spe_sens = calc_spe_sens(exp_clusters, preds)
    open(os.path.join(CUR_DIR, '{}-spe-sens.txt'.format(exp)), 'w').write(
        'Specificity: {:.1f}%\tSensitivity: {:.1f}%\tPrecision: {:.1f}%'.format(*spe_sens)
    )


def main():
    process('sc03-it30-oth1', 'SC02v2', 'SC03')
    process('sc03-it50-oth4', 'SC02v2', 'SC03')
    #process('sc03-it50-oth4', threshold=0.134)
    process('sc03-it50-oth4-pro', 'SC02v2', 'SC03')
    process('sc03-it100-oth2', 'SC02v2', 'SC03')
    process('sc03-it200-cum2', 'SC02v2', 'SC03')
    process('sc03-it200-cum4', 'SC02v2', 'SC03')
    process('sc03-it200-int2', 'SC02v2', 'SC03')
    process('sc03-it200-int4', 'SC02v2', 'SC03')


if __name__ == '__main__':
    main()
