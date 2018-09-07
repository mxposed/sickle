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
    X, _ = utils.load_10x(path('SC02'), 'SC02v2')
    best_model = catboost.CatBoostClassifier(
        l2_leaf_reg=7,
        learning_rate=0.622,
        depth=10,
        iterations=200,
        random_seed=42,
        logging_level='Silent',
        loss_function='MultiClass',
        eval_metric='TotalF1',
        thread_count=20,
    )
    model_path = os.path.join(CUR_DIR, 'sc02v2-best-model.cbm')
    best_model.load_model(model_path)

    sc01, _ = utils.load_10x(path('SC01'), 'SC01v2')
    sc01_preds = predict(best_model, X.columns, sc01)
    sc01_preds.to_csv(os.path.join(CUR_DIR, 'sc01-preds.csv'))


if __name__ == '__main__':
    main()
