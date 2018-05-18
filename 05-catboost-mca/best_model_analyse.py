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


CUR_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def load_predictions(path):
    clusters = pd.read_csv(os.path.join(CUR_DIR, 'MCA_clusters.csv'))
    preds = pd.read_csv(path, index_col=0)
    preds.columns = clusters.Annotation
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


def second_maxes(predictions):
    second_choices = pd.DataFrame(index=predictions.columns, columns=predictions.columns)
    second_choices.fillna(0, inplace=True)
    for cell in predictions.index:
        cell_type = predictions.loc[cell, :].idxmax()
        score = predictions.loc[cell, cell_type]
        second_cell_type = predictions.loc[cell, :][predictions.loc[cell, :] < score].idxmax()
        second_choices.loc[cell_type, second_cell_type] += 1
    return second_choices


def process(exp):
    sc01 = load_predictions(os.path.join(CUR_DIR, '{}-best-preds.csv'.format(exp)))
    sc01_second_maxes = second_maxes(sc01)
    sc01_second_maxes.to_csv(os.path.join(CUR_DIR, '{}-best-second.csv'.format(exp)))
    sc01 = heatmap(sc01)
    sc01.savefig(os.path.join(CUR_DIR, '{}-best-heatmap.png'.format(exp)))


def main():
    process('sc01')
    process('sc02')
    process('sc03')


if __name__ == '__main__':
    main()
