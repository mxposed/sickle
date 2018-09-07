## Model preparation for SC03

### Code

`params_search.py`: code to search for best params to train SC03 model

`train_predict.py`: code to train model with best parameters from the previous step and predict SC01 and SC02. Result is in `sc*-preds.csv`

`analyse.py`: code to score and plot predictions from previous steps

`plot_plasma_predictions.py`: code to plot cells that got predicted as _Plasma cells_ from SC01 and SC02. Figure 7

### Cached files

`sc*-preds.csv`: predictions of SC01 and SC02
