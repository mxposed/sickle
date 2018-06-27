import pandas as pd


def split(X, y):
    result = []
    unique_classes = y.unique()
    for cls in unique_classes:
        cls_idx = (y == cls).index
        X_cls = X.loc[cls_idx, :]
        X_other = X[y != cls, :]
        y_cls = y.loc[cls_idx]
        y_other = y[y != cls]
        other_sample = y_other.groupby(y_other).apply(
            lambda x: x.sample(len(y_cls) // len(y_other.unique()))
        )
        other_sample.index = other_sample.index.droplevel('cluster')
        other_idx = other_sample.index
        X_cls = pd.concat([X_cls, X_other.loc[other_idx, :]])
        y_cls = pd.concat([y_cls, y_other.loc[other_idx]])
        y_cls[y_cls != cls] = 0
        y_cls[y_cls == cls] = 1
        result.append((cls, X_cls, y_cls))
    return result
