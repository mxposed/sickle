import os

import pandas as pd
import catboost

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def path(*args):
    return os.path.join(ROOT, *args)


def predict(model, input_columns, experiment):
    missing_columns = input_columns[~ input_columns.isin(experiment.columns)]
    experiment = experiment.copy()
    experiment[list(missing_columns)] = pd.DataFrame([[0] * len(missing_columns)], index=experiment.index)
    return pd.DataFrame(model.predict_proba(experiment[input_columns]), index=experiment.index)


def main():
    runs = []
    for i in range(1, 6):
        runs.append(
            pd.read_csv(os.path.join(
                CUR_DIR,
                './search-{}-of-5.csv'.format(i)
            ),
            index_col=0)
        )
    runs = pd.concat(runs, ignore_index=True)
    best_row = runs.score.idxmax(axis=0)
    best_params = eval(runs.params[best_row])

    X, y = utils.load_10x(path('SC01'), 'SC01v2')
    best_model = catboost.CatBoostClassifier(
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
    model_path = os.path.join(CUR_DIR, 'sc01-model.cbm')
    if os.path.exists(model_path):
        best_model.load_model(model_path)
    else:
        best_model.fit(
            X, y, [X.shape[1] - 1]
        )
        best_model.save_model(model_path)
        importances = pd.DataFrame(best_model._feature_importance, X.columns)
        importances[importances[0] > 0].sort_values(
            0,
            ascending=False
        ).to_csv(os.path.join(CUR_DIR, 'sc01-features.csv'))

    sc02, _ = utils.load_10x(path('SC02'), 'SC02v2')
    sc02_preds = predict(best_model, X.columns, sc02)
    sc02_preds.to_csv(os.path.join(CUR_DIR, 'sc02-preds.csv'))


if __name__ == '__main__':
    main()
