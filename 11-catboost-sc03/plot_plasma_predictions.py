import sickle

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy.api as sc
import seaborn


CUR_DIR, ROOT = sickle.paths(__file__)


def get_normalised_jchain(exp, annotation):
    cache_file = os.path.join(CUR_DIR, '{}-jchain.csv'.format(exp.lower()))
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, header=None, squeeze=True)

    data = sickle.load_sc_scanpy(exp, annotation)
    data = data[data.obs['n_genes'] < 4000, :]
    data = data[data.obs['n_genes'] > 300, :]
    data = data[data.obs['percent_mito'] < 0.1, :]
    sc.pp.normalize_per_cell(data, counts_per_cell_after=1e4)
    sc.pp.log1p(data)
    jchain = pd.Series(list(data[:, 'Jchain'].X), index=data.obs_names)
    jchain.to_csv(cache_file)
    return jchain


def plot(exp, annotation, preds, figsize=(7, 2.5), subfig=None):
    jchain = get_normalised_jchain(exp, annotation)
    preds = sickle.load_predictions(os.path.join(CUR_DIR, preds), 'SC03')
    preds.columns = preds.columns.str.replace('classical/', 'classical +\n')
    cluster = preds.idxmax(axis=1)
    jchain_expr = pd.DataFrame(
        index=jchain.index,
        columns=preds.columns
    )
    jchain_expr.fillna(0, inplace=True)
    for column in preds.columns:
        idx = cluster.index[cluster == column]
        jchain_expr.loc[idx, column] = jchain.loc[idx]

    jchain_expr = jchain_expr.loc[jchain_expr.sum(axis=1) > 2, :]
    jchain_expr = jchain_expr.loc[:, jchain_expr.sum(axis=0) > 2]
    sorted_expr = jchain_expr.sum(axis=1).sort_values()
    jchain_expr = jchain_expr.reindex(sorted_expr.index)

    more_than_5_row = sorted_expr.index[sorted_expr >= 5][0]
    more_than_5_col = jchain_expr.loc[more_than_5_row, :].idxmax()
    less_than_5_row = sorted_expr.index[sorted_expr < 5][-1]
    less_than_5_col = jchain_expr.loc[less_than_5_row, :].idxmax()
    annot = pd.DataFrame(index=jchain_expr.index, columns=jchain_expr.columns)
    annot.fillna('', inplace=True)
    annot.loc[more_than_5_row, more_than_5_col] = '{:.1f}'.format(
        jchain_expr.loc[more_than_5_row, more_than_5_col]
    )
    annot.loc[less_than_5_row, less_than_5_col] = '{:.1f}'.format(
        jchain_expr.loc[less_than_5_row, less_than_5_col]
    )

    fig = plt.figure(figsize=figsize)
    grid_kws = {"height_ratios": (.9, .05), "hspace": .5}
    ax, cbar_ax = fig.subplots(2, gridspec_kw=grid_kws)

    ax = seaborn.heatmap(
        jchain_expr.T,
        xticklabels=[],
        cmap="Oranges",
        linewidths=1,
        annot=annot.T,
        fmt='',
        annot_kws={
            'fontsize': 10,
            'color': 'white',
            'weight': 'bold',
        },
        ax=ax,
        cbar_ax = cbar_ax,
        cbar_kws={
            'orientation': 'horizontal',
            'ticks': [0, 2.5, 5, 7],
            'label': '$\it{log(normalise(}$Jchain$\it{))}$'
        }
    )
    ax.figure.axes[-1].tick_params(reset=True)
    ax.figure.axes[-1].tick_params(
        labelsize=9,
        direction='inout',
        color='black',
        width=.5,
        length=6,
        pad=2,
        top=False,
        labeltop=False,
    )
    ax.figure.axes[-1].xaxis.label.set_size(14)
    ax.set_xlabel('Cells from {} dataset'.format(exp))
    ax.set_ylabel('')
    plt.tight_layout()
    if subfig:
        text_ax = ax.figure.add_axes((0.02, 0.1, 0.05, 0.05), frame_on=False)
        text_ax.set_axis_off()
        plt.text(0, 0, subfig, fontsize=30, transform=text_ax.transAxes, weight='black')
    ax.figure.subplots_adjust(left=0.3, top=1, right=0.98, bottom=0.20)
    ax.figure.savefig(os.path.join(CUR_DIR, '{}-jchain.png'.format(exp.lower())))


def main():
    plot('SC01', 'SC01v2', 'sc01-preds.csv', subfig='A')
    plot('SC02', 'SC02v2', 'sc02-preds.csv', subfig='B')


if __name__ == '__main__':
    main()
