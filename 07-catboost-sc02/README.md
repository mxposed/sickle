## Model preparation for SC02

### Code

`find_best_params_v2.py`: code to search for best params to train SC02 model (SC02v2 clustering)

`best_model_predict_v2.py`: code to train model with best parameters from the previous step and predict SC03. Result is in `sc03v2-preds.csv`

`predict_sc01.py`: code to dump predict SC01. Result is in `sc01-preds.csv`

`best_model_analyse_v2.py`: code to score and plot predictions from previous steps. Figure 4A, Figure 8

### Cached files

`sc*-preds.csv`: predictions by best SC02 model
