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
    return exp


def cross_val(X, y, params, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True)

    scores = []

    for tr_ind, val_ind in skf.split(X, y):
        X_train, y_train = X.iloc[tr_ind, :], y.iloc[tr_ind]
        X_valid, y_valid = X.iloc[val_ind, :], y.iloc[val_ind]

        clf = catboost.CatBoostClassifier(
            l2_leaf_reg=params['l2_leaf_reg'],
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            iterations=30,
            random_seed=42,
            logging_level='Silent',
            loss_function='MultiClass',
            eval_metric='TotalF1',
            #thread_count=20,
        )

        clf.fit(
            X_train,
            y_train,
            cat_features=[X.shape[1] - 1],
        )

        y_pred = clf.predict(X_valid)
        score = sklearn.metrics.f1_score(y_valid, y_pred, average='weighted')
        print(params)
        print('Score: {}'.format(score))
        print('-'*20)
        scores.append(score)
    return np.mean(scores)


def catboost_GridSearchCV(X, y, params_space, cv=5):
    max_score = 0
    best_params = None
    for params in list(ParameterGrid(params_space)):
        score = cross_val(X, y, params, cv=cv)
        if score > max_score:
            max_score = score
            best_params = params
    return best_params


def main():
    sc02 = load_10x(path('SC02'), 'SC02')
    X, y = sc02[sc02.columns[:-1]], sc02.cluster
    X_train, X_test, y_train, y_test =  train_test_split(
        X, y, test_size=0.1, stratify=y,
    )
    params_space = {
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'learning_rate': np.linspace(1e-3, 8e-1, num=10),
        'depth': [6, 8, 10],
    }
    best_params = catboost_GridSearchCV(X_train, y_train, params_space, cv=5)
    print(best_params)


if __name__ == '__main__':
    main()
