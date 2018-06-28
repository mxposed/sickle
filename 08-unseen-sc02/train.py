import catboost
import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def train(classes):
    result = []
    for cls, X, y in classes:
        model = catboost.CatBoostClassifier(
            l2_leaf_reg=2,
            learning_rate=0.622,
            depth=10,
            iterations=30,
            random_seed=42,
            logging_level='Silent',
            loss_function='Logloss',
            eval_metric='F1',
            thread_count=20,
        )
        model.fit(X, y)
        result.append((cls, model))
    return result


def save(models):
    model_dir = os.path.join(CUR_DIR, 'models')
    for cls, model in models:
        model_path = os.path.join(model_dir, 'cls-{}.cbm'.format(cls))
        model.save_model(model_path)


def models(classes):
    model_dir = os.path.join(CUR_DIR, 'models')
    result = []
    for cls, _, _ in classes:
        model = catboost.CatBoostClassifier(
            l2_leaf_reg=2,
            learning_rate=0.622,
            depth=10,
            iterations=30,
            random_seed=42,
            logging_level='Silent',
            loss_function='Logloss',
            eval_metric='F1',
            thread_count=20,
        )
        model_path = os.path.join(model_dir, 'cls-{}.cbm'.format(cls))
        model.load_model(model_path)
        result.append((cls, model))
    return result
