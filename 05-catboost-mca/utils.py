import os

import pandas as pd


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

    lung = lung.join(cell_types.ClusterID)
    lung = lung[~lung.ClusterID.isna()]
    lung.fillna(0, inplace=True)

    X = lung[lung.columns[lung.columns != 'ClusterID']]
    y = lung.ClusterID
    y = y.str.replace('Lung_', '')
    y = y.astype('int')
    y = y - 1
    return X, y
