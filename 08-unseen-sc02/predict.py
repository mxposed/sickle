import pandas as pd


def base_predict(model, input_columns, experiment):
    missing_columns = input_columns[~ input_columns.isin(experiment.columns)]
    experiment = experiment.copy()
    experiment[list(missing_columns)] = pd.DataFrame([[0] * len(missing_columns)], index=experiment.index)
    return pd.Series(model.predict_proba(experiment[input_columns])[:,1], index=experiment.index)


def predict(models, input_columns, X):
    result = pd.DataFrame(index=X.index, columns=[cls for cls, _ in models])
    for cls, model in models:
        result[cls] = base_predict(model, input_columns, X)
    return result
