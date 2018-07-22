import os

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scanpy.api as sc


CUR_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def load_10x_scanpy(path, batch_label):
    sc01 = sc.read('{}/matrix.mtx'.format(path), cache=True).T
    sc01.var_names = pd.read_table('{}/genes.tsv'.format(path), header=None)[1]
    sc01.obs_names = pd.read_table('{}/barcodes.tsv'.format(path), header=None)[0]
    sc01.obs_names = sc01.obs_names.str.replace('-1', '')
    sc01.var_names_make_unique()
    sc.pp.filter_cells(sc01, min_genes=200)
    sc.pp.filter_genes(sc01, min_cells=3)

    mito_genes = sc01.var_names[sc01.var_names.str.match(r'^mt-')]
    sc01.obs['n_UMI'] = np.sum(sc01.X, axis=1).A1
    sc01.obs['percent_mito'] = np.sum(sc01[:, mito_genes].X, axis=1).A1 / sc01.obs['n_UMI']

    assgn = pd.read_csv('{}/{}_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
        batch_label,
    ), index_col=0)
    assgn.columns = ['cluster']

    sc01.obs['cluster'] = assgn.cluster[sc01.obs.index]
    return sc01


def load_10x(path, batch_label):
    sc01 = load_10x_scanpy(path, batch_label)
    exp = pd.DataFrame(sc01.X.todense(), index=sc01.obs_names, columns=sc01.var_names)
    exp['Batch'] = batch_label
    exp.fillna(0, inplace=True)

    exp['cluster'] = sc01.obs.cluster[exp.index]
    exp = exp[~exp.cluster.isna()]
    return exp[exp.columns[:-1]], exp.cluster


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
