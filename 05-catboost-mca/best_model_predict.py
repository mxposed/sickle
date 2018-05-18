import pandas as pd
import numpy as np
import sklearn
import catboost
import scipy.sparse
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import seaborn
import os.path


CUR_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def path(*args):
    return os.path.join(ROOT, *args)

def load_mca_lung():
    lung1 = pd.read_csv(path('rmbatch_dge/Lung1_rm.batch_dge.txt'), header=0, sep=' ', quotechar='"')
    lung2 = pd.read_csv(path('rmbatch_dge/Lung2_rm.batch_dge.txt'), header=0, sep=' ', quotechar='"')
    lung3 = pd.read_csv(path('rmbatch_dge/Lung3_rm.batch_dge.txt'), header=0, sep=' ', quotechar='"')

    lung1 = lung1.transpose()
    lung2 = lung2.transpose()
    lung3 = lung3.transpose()
    lung = pd.concat([lung1, lung2, lung3])

    cell_types = pd.read_csv(path('MCA_assign.csv'), index_col=0)
    cell_types = cell_types[cell_types.Tissue == 'Lung']
    cell_types = cell_types.set_index('Cell.name')
    cell_types = cell_types[['ClusterID', 'Batch']]

    lung = lung.join(cell_types)
    lung = lung[~lung.Batch.isna()]
    lung.fillna(0, inplace=True)

    X = lung[lung.columns[lung.columns != 'ClusterID']]
    y = lung.ClusterID
    y = y.str.replace('Lung_', '')
    y = y.astype('int')
    y = y - 1
    return X, y

def load_10x(path, batch_label):
    mtx = pd.read_csv('{}/matrix.mtx'.format(path), skiprows=3, sep=' ', header=None)
    genes = pd.read_table('{}/genes.tsv'.format(path), header=None, index_col=1)
    cells = pd.read_table('{}/barcodes.tsv'.format(path), header=None, index_col=0)
    exp = scipy.sparse.csc_matrix((mtx[2], (mtx[0] - 1, mtx[1] - 1)), shape=(len(genes), len(cells)))
    exp = pd.SparseDataFrame(exp)
    exp.columns = cells.index
    exp = exp.transpose()
    exp.columns = genes.index
    exp = exp.to_dense()
    exp['Batch'] = batch_label
    exp.fillna(0, inplace=True)
    exp = exp.groupby(level=0, axis=1).sum()
    return exp

def predict(model, input_columns, experiment):
    missing_columns = input_columns[~ input_columns.isin(experiment.columns)]
    experiment = experiment.copy()
    experiment[list(missing_columns)] = pd.DataFrame([[0] * len(missing_columns)], index=experiment.index)
    return model.predict_proba(experiment[input_columns])

def main():
    X, y = load_mca_lung()
    best_model = catboost.CatBoostClassifier(
        l2_leaf_reg=2,
        learning_rate=0.415,
        depth=8,
        iterations=200,
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='Accuracy',
    )
    best_model.fit(
        X, y, [X.shape[1] - 1]
    )
    best_model.save_model(os.path.join(CUR_DIR, 'mca-lung-best-model.cbm'))

    sc01 = load_10x(path('SC01'), 'SC01')
    sc02 = load_10x(path('SC02'), 'SC02')
    sc03 = load_10x(path('SC03'), 'SC03')

    predictions01 = predict(best_model, X.columns, sc01)
    predictions02 = predict(best_model, X.columns, sc02)
    predictions03 = predict(best_model, X.columns, sc03)

    pd.DataFrame(predictions01).to_csv(os.path.join(CUR_DIR, 'sc01-best-preds.csv'))
    pd.DataFrame(predictions02).to_csv(os.path.join(CUR_DIR, 'sc02-best-preds.csv'))
    pd.DataFrame(predictions03).to_csv(os.path.join(CUR_DIR, 'sc03-best-preds.csv'))

if __name__ == '__main__':
    main()
