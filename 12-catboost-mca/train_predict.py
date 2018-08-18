import sickle

import os
import sys

import catboost
import pandas as pd


CUR_DIR, ROOT = sickle.paths(__file__)


def predict(model, input_columns, experiment):
    missing_columns = input_columns[~ input_columns.isin(experiment.columns)]
    experiment = experiment.copy()
    experiment[list(missing_columns)] = pd.DataFrame([[0] * len(missing_columns)], index=experiment.index)
    return pd.DataFrame(model.predict_proba(experiment[input_columns]), index=experiment.index)


def train(splits):
    runs = []
    for i in range(1, splits + 1):
        runs.append(
            pd.read_csv(os.path.join(
                CUR_DIR,
                './search-{}-of-{}.csv'.format(i, splits)
            ),
            index_col=0)
        )
    runs = pd.concat(runs, ignore_index=True)
    best_row = runs.score.idxmax(axis=0)
    best_params = eval(runs.params[best_row])

    X, y = sickle.load_mca_lung('MCAv2')
    model = catboost.CatBoostClassifier(
        l2_leaf_reg=best_params['l2_leaf_reg'],
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth'],
        iterations=200,
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='TotalF1',
        thread_count=20,
    )
    model_path = os.path.join(CUR_DIR, 'mca-model.cbm')
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        model.fit(X, y)
        model.save_model(model_path)
        importances = pd.DataFrame(model._feature_importance, X.columns)
        importances[importances[0] > 0].sort_values(
            0,
            ascending=False
        ).to_csv(os.path.join(CUR_DIR, 'mca-features.csv'))
    return model, X.columns


def main(splits):
    model, input_columns = train(splits)

    for exp in ('SC01v2', 'SC02v2', 'SC03'):
        data, _ = sickle.load_sc(exp)
        preds = predict(model, input_columns, data)
        preds.to_csv(os.path.join(
            CUR_DIR,
            '{}-preds.csv'.format(exp[:4].lower())
        ))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <N>'.format(__file__), file=sys.stderr)
        sys.exit(1)
    splits = int(sys.argv[1])
    main(splits)
