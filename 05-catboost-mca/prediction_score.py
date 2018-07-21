import os

import catboost
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

import utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CUR_DIR))


def process():
    predictions = pd.read_csv(os.path.join(CUR_DIR, 'sc01-best-preds.csv'))
    y = predictions.max(axis=1)
    X, _ = utils.load_10x(os.path.join(ROOT, 'SC01'), 'SC01v2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = catboost.CatBoostRegressor(
        iterations=200,
        learning_rate=0.4,
        depth=10,
        l2_leaf_reg=3,
        random_seed=42,
        logging_level='Silent',
        thread_count=20,
        loss_function='MAE',
    )
    model.fit(X_train, y_train)
    model.save_model(os.path.join(CUR_DIR, 'sc01-pred-score.cbm'))
    y_pred = model.predict(X_test)
    score = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    print('MAE test score: {}'.format(score))

    importances = pd.DataFrame(model._feature_importance, X.columns)
    importances[importances[0] > 0].sort_values(
        0,
        ascending=False
    ).to_csv(os.path.join(CUR_DIR, 'sc01-pred-score-features.csv'))


def main():
    process()


if __name__ == '__main__':
    main()
