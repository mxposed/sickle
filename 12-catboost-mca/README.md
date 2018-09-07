## Model preparation for MCA

### Code

`params_search.py`: code to search for best params to train MCA model (our clustering)

`train_predict.py`: code to train model with best parameters from the previous step and predict SC01, SC02 and SC03. Result is in `sc*-preds.csv`

`analyse.py`: code to score and plot predictions from previous steps. Figure 5

`prediction_correlation.py`: code to test correlation between technical variables and prediction probability of the model on SC01. Table 2, Figure 6

### Cached files

`sc*-preds.csv`: predictions of SC01, SC02 and SC03
