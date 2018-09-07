import sickle

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn

import sankey
import utils


CUR_DIR, ROOT = sickle.paths(__file__)


def heatmap(predictions, figsize=(16, 20)):
    preds = predictions.copy()
    preds.columns = list(range(len(preds.columns)))
    preds['cluster'] = preds.idxmax(axis=1)
    preds['max_score'] = predictions.max(axis=1)
    preds.sort_values(['cluster', 'max_score'], ascending=[True, False], inplace=True)
    preds = predictions.reindex(preds.index)
    plt.figure(figsize=figsize)
    ax = seaborn.heatmap(preds, yticklabels=[], rasterized=True)
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def process(exp, reference, query, tag=None):
    exp_clusters = sickle.assignments(query)

    clusters = sickle.load_clusters(query)

    preds = sickle.load_predictions(
        os.path.join(CUR_DIR, '{}-preds.csv'.format(exp)),
        reference
    )

    hmap = heatmap(preds)
    hmap.savefig(os.path.join(CUR_DIR, '{}-heatmap.pdf'.format(exp)))

    seaborn.set(font_scale=1)
    mapping = sickle.mapping(query, reference)
    s = sankey.sankey(
        clusters.iloc[:, 0].loc[exp_clusters],
        preds.idxmax(axis=1),
        alpha=.7,
        left_order=sickle.sankey_order(),
        mapping=mapping,
        tag=tag
    )
    s.savefig(os.path.join(CUR_DIR, '{}-sankey.pdf'.format(exp)))

    if mapping:
        open(os.path.join(CUR_DIR, '{}-f1.txt'.format(exp)), 'w').write(
            'F1 score: {:.4f}'.format(
                mapping.f1(clusters.iloc[:, 0].loc[exp_clusters],
                           preds.idxmax(axis=1))
            )
        )


def main():
    process('sc01', 'MCA', 'SC01v2', tag='A')
    process('sc02', 'MCA', 'SC02v2', tag='B')
    process('sc03', 'MCA', 'SC03')


if __name__ == '__main__':
    main()
