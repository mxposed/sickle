import math

import pandas as pd


def split(X, y, other_proportion=1, splits=None, split_order=None, other=None):
    if other not in ('equal', 'proportional'):
        raise ValueError('other should be equal or proportional')
    result = []
    class_splits = split_y(y, splits=splits, split_order=split_order)
    for classes in class_splits:
        membership = y.isin(classes)
        cls_idx = y[membership].index
        inverse_idx = y[~membership].index
        X_cls = X.loc[cls_idx, :]
        X_other = X.loc[inverse_idx, :]
        y_cls = y.loc[cls_idx]
        y_other = y.loc[inverse_idx]

        y_cls = y_cls.replace({x: i for i, x in enumerate(classes)})

        def to_take(x):
            if other == 'equal':
                return len(y_cls) * other_proportion // len(y_other.unique())
            else:
                total = len(y_cls) * other_proportion
                class_frac = (y == x[0]).sum() / len(y)
                return total * class_frac
        other_sample = y_other.groupby(y_other).apply(
            lambda x: x.sample(to_take(x), replace=len(x) < to_take)
        )
        other_sample.index = other_sample.index.droplevel('cluster')
        other_idx = other_sample.index
        X_cls = pd.concat([X_cls, X_other.loc[other_idx, :]])
        y_cls = pd.concat([y_cls, pd.Series(len(classes), index=other_idx)])
        result.append((classes, X_cls, y_cls))
    return result


def split_y(y, other_proportion=1, splits=None, split_order=None):
    result = []
    unique_classes = sorted(y.unique())
    if splits is None:
        splits = len(unique_classes)
    if splits > len(unique_classes):
        raise ValueError('number of splits too large, max {}'.format(
            len(unique_classes)
        ))
    if split_order not in ('cumsum', 'interleaved'):
        raise ValueError('split_order should be cumsum or interleaved')
    max_classes_in_split = math.ceil(len(unique_classes) / splits)
    max_items_in_split = len(y) / splits

    order_idx = list(range(len(unique_classes)))
    if split_order == 'interleaved':
        order = []
        for i in range(splits):
            order += order_idx[i::splits]
        order_idx = order

    current_split_classes = []
    current_sum = 0
    for cls, count in y.value_counts().sort_index()[order_idx].iteritems():
        current_split_classes.append(cls)
        current_sum += count
        if len(result) + 1 < splits \
            and (current_sum >= max_items_in_split \
                or len(current_split_classes) >= max_classes_in_split):
            result.append(current_split_classes)
            current_split_classes = []
            current_sum = 0
    if len(current_split_classes):
        result.append(current_split_classes)
    return result
