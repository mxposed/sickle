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
    clusters = pd.read_csv('{}/SC02v2_clusters.csv'.format(
        os.path.join(CUR_DIR, '..', '00-metadata'),
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(int)
    preds = pd.read_csv(path, index_col=0)
    preds.columns = clusters.iloc[:,0]
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


def process(exp):
    preds = load_predictions(os.path.join(CUR_DIR, '{}-preds.csv'.format(exp)))
    hmap = heatmap(preds)
    hmap.suptitle('Predictions for SC03 dataset')
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-heatmap.png'.format(exp)))

    sc03_clusters = pd.read_csv('{}/SC03_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
    ), index_col=0)
    sc03_clusters.columns = ['cluster']
    hmap = heatmap(preds.loc[sc03_clusters.index[sc03_clusters.cluster == 7],:], figsize=(16, 12))
    hmap.suptitle('Predictions for SC03 Plasma cells')
    hmap.subplots_adjust(top=0.88)
    hmap.savefig(os.path.join(CUR_DIR, '{}-plasma-heatmap.png'.format(exp)))


def main():
    process('sc03v2')


if __name__ == '__main__':
    main()
