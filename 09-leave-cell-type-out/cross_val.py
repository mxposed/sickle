import pandas as pd
from sklearn.model_selection import train_test_split


def leave_cell_type_out(X, y):
    for cls in sorted(y.unique()):
        y_train = y[y == cls]
        y_test = y[y != cls]
        X_train = X.loc[y_train.index, :]
        X_test = X.loc[y_test.index, :]
        X_train, X_test_add, y_train, y_test_add =  train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train,
        )
        X_test = pd.concat([X_test, X_test_add])
        y_test = pd.concat([y_test, y_test_add])
        yield X_train, y_train, X_test, y_test
