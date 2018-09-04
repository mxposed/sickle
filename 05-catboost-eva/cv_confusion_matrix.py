import os

import pandas as pd

import utils
import seaborn
import matplotlib.pyplot as plt


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def results_matrix(x, y, x_cols=None, y_cols=None):
    if x_cols is None:
        x_cols = sorted(x.unique())
    if y_cols is None:
        y_cols = sorted(y.unique())
    res = pd.DataFrame(index=x_cols, columns=y_cols)
    for i in res.index:
        res.loc[i, :] = y[(x[x == i]).index].value_counts(sort=False).sort_index()
    res.fillna(0, inplace=True)
    sums = res.sum(axis=1)
    return res.div(sums, axis=0)


def draw(cv_num):
    mca_clusters = pd.read_csv(
        os.path.join(CUR_DIR, 'MCA_clusters.csv'),
        index_col=1,
    )
    mca_clusters.index = mca_clusters.index.astype(int) - 1
    y_test = utils.load_mca_assignments()
    predictions = pd.read_csv(
        os.path.join(CUR_DIR, 'cv{}-predictions.csv'.format(cv_num)),
        index_col=0,
    )
    predictions.columns = mca_clusters.Annotation
    y_pred = predictions.idxmax(axis=1)
    y_test = mca_clusters.Annotation[y_test[y_pred.index]]
    y_test.index = y_pred.index

    confusion = results_matrix(y_test, y_pred, x_cols=mca_clusters.Annotation, y_cols=mca_clusters.Annotation)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cbar_ax = fig.add_axes((0.08, 0.2, 0.2, 0.02))
    ax = seaborn.heatmap(
        confusion,
        square=True,
        ax=ax,
        cmap="Blues",
        cbar_ax=cbar_ax,
        cbar_kws={
            'orientation': 'horizontal',
        }
    )
    ax.figure.axes[-1].tick_params(direction='inout', length=10)
    ax.figure.axes[-1].set_xlabel('Fraction of cells in row', fontsize=13)
    ax.set_xlabel('Predicted cell type', fontsize=16)
    ax.set_ylabel('Annotated cell type', fontsize=16)

    plt.tight_layout()
    ax.figure.savefig(os.path.join(CUR_DIR, 'cv{}-confusion.png'.format(cv_num)))


def main():
    draw(3)


if __name__ == '__main__':
    main()
