import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import scanpy.api as sc

import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


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


def load_predictions(path, reference):
    clusters = pd.read_csv('{}/{}_clusters.csv'.format(
        os.path.join(os.path.dirname(CUR_DIR), '00-metadata'),
        reference
    ), index_col=0, header=None)
    clusters.index = clusters.index.str.replace('C', '').astype(int)
    preds = pd.read_csv(path, index_col=0)
    preds.columns = clusters.iloc[:,0]
    return preds
