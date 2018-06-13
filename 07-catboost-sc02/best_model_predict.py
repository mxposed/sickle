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
    X, y = utils.load_10x(path('SC02'), 'SC02')
    best_model = catboost.CatBoostClassifier(
        l2_leaf_reg=3,
        learning_rate=0.445,
        depth=10,
        iterations=200,
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='TotalF1',
        thread_count=20,
    )
    model_path = os.path.join(CUR_DIR, 'sc02-best-model.cbm')
    if os.path.exists(model_path):
        best_model.load_model(model_path)
    else:
        best_model.fit(
            X, y, [X.shape[1] - 1]
        )
        best_model.save_model(model_path)

    sc03, _ = utils.load_10x(path('SC03'), 'SC03')
    sc03_preds = predict(best_model, X.columns, sc03)
    sc03_preds.to_csv(os.path.join(CUR_DIR, 'sc03-preds.csv'))

    importances = pd.DataFrame(best_model._feature_importance, X.columns)
    importances[importances[0] > 0].sort_values(
        0,
        ascending=False
    ).to_csv(os.path.join(CUR_DIR, 'sc02-model-features.csv'))


if __name__ == '__main__':
    main()
