import catboost
import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model(classes=classes, iterations=30):
    if len(classes) == 1:
        loss_function = 'Logloss'
    else:
        loss_function = 'MultiClass'
    model = catboost.CatBoostClassifier(
        l2_leaf_reg=7,
        learning_rate=0.622,
        depth=10,
        iterations=iterations,
        random_seed=42,
        logging_level='Silent',
        loss_function=loss_function,
        eval_metric='F1',
        thread_count=20,
    )
    return model


def models(classes, iterations=30, label=None):
    model_dir = os.path.join(CUR_DIR, 'models', label)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    result = []
    for cls, X, y in classes:
        model = get_model(classes=cls, iterations=iterations)
        model_path = os.path.join(model_dir, 'cls-{}.cbm'.format('+'.join(cls)))
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            model.fit(X, y)
            model.save_model(model_path)
        result.append((cls, model))
    return result
