import os
import sys

import catboost
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def train(params, X_train, y_train):
    clf = catboost.CatBoostClassifier(
        l2_leaf_reg=params['l2_leaf_reg'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        iterations=200,
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='TotalF1',
        thread_count=20,
    )

    clf.fit(
        X_train,
        y_train,
    )
    return clf


def main(number):
    runs = []
    for i in range(1, 11):
        runs.append(
            pd.read_csv(os.path.join(
                CUR_DIR,
                './search-{}+{}-of-10.csv'.format(number, i)
            ),
            index_col=0)
        )
    runs = pd.concat(runs, ignore_index=True)
    best_row = runs.score.idxmax(axis=0)
    best_params = eval(runs.params[best_row])

    X, y = utils.load_mca_lung()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    for tr_idx, test_idx in splits[number - 1:number]:
        X_train, y_train = X.iloc[tr_idx, :], y.iloc[tr_idx]
        X_test, y_text = X.iloc[test_idx, :], y.iloc[test_idx]

        model = train(best_params, X_train, y_train)
        model.save_model(os.path.join(CUR_DIR, 'model-cv{}.cbm'.format(number)))

        importances = pd.DataFrame(model._feature_importance, X_train.columns)
        importances[importances[0] > 0].sort_values(
            0,
            ascending=False
        ).to_csv(os.path.join(CUR_DIR, 'model-cv{}-features.csv'.format(number)))
        predictions = pd.DataFrame(model.predict_proba(X_test), index=X_test.index)
        predictions.to_csv(os.path.join(CUR_DIR, 'cv{}-predictions.csv'.format(number)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <N>'.format(__file__), file=sys.stderr)
        sys.exit(1)
    main(int(sys.argv[1]))

