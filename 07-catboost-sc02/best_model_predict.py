import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
import catboost
import sklearn.metrics

import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def path(*args):
    return os.path.join(ROOT, *args)


def load_10x(path, batch_label):
    mtx = pd.read_csv('{}/matrix.mtx'.format(path), skiprows=3, sep=' ', header=None)
    genes = pd.read_table('{}/genes.tsv'.format(path), header=None, index_col=1)
    cells = pd.read_table('{}/barcodes.tsv'.format(path), header=None, index_col=0)
    assgn = pd.read_csv('{}/{}_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
        batch_label,
    ), index_col=0)
    assgn.columns = ['cluster']
    exp = scipy.sparse.csc_matrix((mtx[2], (mtx[0] - 1, mtx[1] - 1)), shape=(len(genes), len(cells)))
    exp = pd.SparseDataFrame(exp)
    exp.columns = cells.index.str.replace('-1', '')
    exp = exp.transpose()
    exp.columns = genes.index
    exp = exp.to_dense()
    exp['Batch'] = batch_label
    exp.fillna(0, inplace=True)
    cols = exp.columns.unique()
    exp = exp.groupby(level=0, axis=1).sum()
    exp = exp.reindex(columns=cols)
    exp = exp.join(assgn)
    exp = exp[~exp.cluster.isna()]
    return exp[exp.columns[:-1]], exp.cluster


def predict(model, input_columns, experiment):
    missing_columns = input_columns[~ input_columns.isin(experiment.columns)]
    experiment = experiment.copy()
    experiment[list(missing_columns)] = pd.DataFrame([[0] * len(missing_columns)], index=experiment.index)
    return pd.DataFrame(model.predict_proba(experiment[input_columns]), index=experiment.index)


def main():
    X, y = load_10x(path('SC02'), 'SC02')
    best_model = catboost.CatBoostClassifier(
        l2_leaf_reg=3,
        learning_rate=0.445,
        depth=10,
        iterations=200,
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='TotalF1',
        #thread_count=20,
    )
    model_path = os.path.join(CUR_DIR, 'sc02-best-model.cbm')
    if os.path.exists(model_path):
        best_model.load_model(model_path)
    else:
        best_model.fit(
            X, y, [X.shape[1] - 1]
        )
        best_model.save_model(model_path)

    sc03, _ = load_10x(path('SC03'), 'SC03')
    sc03_preds = predict(best_model, X.columns, sc03)
    pd.DataFrame(sc03_preds).to_csv(os.path.join(CUR_DIR, 'sc03-preds.csv'))


if __name__ == '__main__':
    main()
