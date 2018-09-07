import sickle

import os

import pandas as pd
import seaborn

import sankey


CUR_DIR, ROOT = sickle.paths(__file__)


def process(exp, reference, query, tag=None):
    exp_clusters = sickle.assignments(query)

    clusters = sickle.load_clusters(query)

    preds = pd.read_csv(
        os.path.join(CUR_DIR, '{}-preds.csv'.format(exp)),
        index_col=0,
    )
    preds = preds.reindex(exp_clusters.index)

    seaborn.set(font_scale=1)
    mapping = sickle.mapping(query, reference)
    s = sankey.sankey(
        clusters.iloc[:, 0].loc[exp_clusters],
        preds.x,
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
                           preds.x)
            )
        )


def main():
    process('sc03-cluster', 'SC02v2', 'SC03', tag='A')
    process('sc03-cell', 'SC02v2', 'SC03')


if __name__ == '__main__':
    main()
