import pandas as pd
import scanpy.api as sc

import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def load_10x(path, batch_label):
    sc01 = sc.read('{}/matrix.mtx'.format(path), cache=True).T
    sc01.var_names = pd.read_table('{}/genes.tsv'.format(path), header=None)[1]
    sc01.obs_names = pd.read_table('{}/barcodes.tsv'.format(path), header=None)[0]
    sc01.var_names_make_unique()
    sc.pp.filter_cells(sc01, min_genes=200)
    sc.pp.filter_genes(sc01, min_cells=3)
    exp = pd.DataFrame(sc01.X.todense(), index=sc01.obs_names, columns=sc01.var_names)
    exp['Batch'] = batch_label
    exp.fillna(0, inplace=True)

    assgn = pd.read_csv('{}/{}_assgn.csv'.format(
        os.path.join(CUR_DIR, '..', '01-cluster-sc01-sc02'),
        batch_label,
    ), index_col=0)
    assgn.columns = ['cluster']
    exp = exp.join(assgn)
    exp = exp[~exp.cluster.isna()]
    return exp[exp.columns[:-1]], exp.cluster
