import pandas as pd


def base_predict(model, input_columns, experiment):
    missing_columns = input_columns[~ input_columns.isin(experiment.columns)]
    experiment = experiment.copy()
    experiment[list(missing_columns)] = pd.DataFrame([[0] * len(missing_columns)], index=experiment.index)
    return pd.DataFrame(model.predict_proba(experiment[input_columns]), index=experiment.index)


def predict(models, input_columns, X):
    result = pd.DataFrame(index=X.index)
    for i, (classes, model) in enumerate(models):
        predictions = base_predict(model, input_columns, X)
        if len(classes) == 1:
            results[classes] = predictions[0]
        else:
            classes += [-i - 1]
            result[classes] = predictions
    return result
