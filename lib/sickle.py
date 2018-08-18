import display_settings

import collections
import os

import numpy as np
import pandas as pd
import scanpy.api as sc
import sklearn.metrics


dirs = {}

def paths(path):
    dirs['cur'] = os.path.dirname(os.path.abspath(path))
    dirs['code'] = os.path.dirname(dirs['cur'])
    dirs['root'] = os.path.dirname(dirs['code'])
    return dirs['cur'], dirs['root']


def sankey_order():
    return pd.read_csv(os.path.join(dirs['code'], 'sankey_order.csv')).order


def assignments(dataset):
    exp_clusters = pd.read_csv('{}/{}_assgn.csv'.format(
        os.path.join(dirs['code'], '01-cluster-sc01-sc02'),
        dataset
    ), index_col=0)
    return exp_clusters.iloc[:, 0]


class Mapping:
    def __init__(self, table):
        self.table = table
        self.src_mult = set()
        self.dst_mult = set()

        for name, count in collections.Counter(table['from']).items():
            if count > 1:
                self.src_mult.add(name)

        for name, count in collections.Counter(table['to']).items():
            if count > 1:
                self.dst_mult.add(name)

        self.src_replace = {}
        self.dst_replace = {}
        for idx in table.index:
            src = table.loc[idx, 'from']
            dst = table.loc[idx, 'to']
            if src in self.src_mult:
                self.dst_replace[dst] = src
            else:
                self.src_replace[src] = dst

    def category(self, source, dest):
        rows_src = self.table[self.table['from'] == source]
        rows = rows_src[rows_src['to'] == dest]
        if len(rows) == 0:
            if len(rows_src) == 1 and rows_src['to'].iloc[0] == 'Novel cell type':
                return 'novel'
            return 'mistake'
        if source in self.src_mult:
            return 'increase'
        if dest in self.dst_mult:
            return 'decrease'
        return 'correct'

    def f1(self, true, pred):
        true = true.replace(self.src_replace)
        pred = pred.replace(self.dst_replace)
        return sklearn.metrics.f1_score(true, pred, average='weighted')


def mapping(query, reference):
    mapping_file = os.path.join(
        dirs['code'],
        '00-metadata',
        '{}_to_{}.csv'.format(query, reference)
    )
    if os.path.exists(mapping_file):
        return Mapping(pd.read_csv(mapping_file, index_col=None))


def load_mca_assignments(annotation):
    if annotation == 'MCAv2':
        assgn = pd.read_csv(
            os.path.join(dirs['code'], '13-cluster-mca', 'MCAv2_assign.csv'),
            index_col=0
        )
        assgn.columns = ['cluster']
        return assgn.cluster.astype(int) - 1
    raise ValueError('Annotation not known')


def load_mca_raw():
    data = []
    for i in range(1, 4):
        batch = pd.read_csv(
            os.path.join(
                dirs['root'],
                'rmbatch_dge',
                'Lung{}_rm.batch_dge.txt'.format(i)
            ),
            header=0,
            sep=' ',
            quotechar='"'
        )
        data.append(batch.transpose())
    return pd.concat(data)


def load_mca_lung(annotation):
    lung = load_mca_raw().join(load_mca_assignments(annotation))
    lung = lung[~lung.cluster.isna()]
    lung.fillna(0, inplace=True)

    X = lung[lung.columns[lung.columns != 'cluster']]
    return X, lung.cluster


def load_sc_scanpy(exp_name, batch_label):
    exp_dir = os.path.join(dirs['root'], exp_name)

    data = sc.read(os.path.join(exp_dir, '/matrix.mtx'), cache=True).T
    data.var_names = pd.read_table(
        os.path.join(exp_dir, 'genes.tsv'),
        header=None
    )[1]
    data.obs_names = pd.read_table(
        os.path.join(exp_dir, 'barcodes.tsv'),
        header=None
    )[0]
    data.obs_names = data.obs_names.str.replace('-1', '')
    data.var_names_make_unique()
    sc.pp.filter_cells(data, min_genes=200)
    sc.pp.filter_genes(data, min_cells=3)

    data.obs['n_UMI'] = np.sum(data.X, axis=1).A1

    mito_genes = data.var_names[data.var_names.str.match(r'^mt-')]
    data.obs['percent_mito'] = np.sum(data[:, mito_genes].X, axis=1).A1 / data.obs['n_UMI']

    ribo_genes = data.var_names[data.var_names.str.match(r'^(Rpl|Rps|Mrpl|Mrps)')]
    data.obs['percent_ribo'] = np.sum(data[:, ribo_genes].X, axis=1).A1 / data.obs['n_UMI']

    assgn = pd.read_csv(os.path.join(
        dirs['code'],
        '01-cluster-sc01-sc02',
        '{}_assgn.csv'.format(batch_label)
    ), index_col=0)
    assgn.columns = ['cluster']

    data.obs['cluster'] = assgn.cluster[data.obs.index]
    return data


def load_sc(batch_label):
    exp_name = batch_label[:4]
    data = load_sc_scanpy(exp_name, batch_label)
    exp = pd.DataFrame(data.X.todense(), index=data.obs_names, columns=data.var_names)
    exp['Batch'] = batch_label
    exp.fillna(0, inplace=True)

    exp['cluster'] = data.obs.cluster[exp.index]
    exp = exp[~exp.cluster.isna()]
    return exp[exp.columns[:-1]], exp.cluster
